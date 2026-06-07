import os
import torch
import torch.nn as nn

from .base_method import Base_method


class KDMethod(Base_method):
    r"""Vanilla Knowledge Distillation method for OpenSTL.

    Core design:
        - Keep the student's native training logic
        - Keep the teacher's native prediction logic
        - Only add an extra KD loss outside

    Final loss:
        total_loss = student_native_loss + kd_weight * kd_loss
    """

    def __init__(self, **args):
        self.student_method = None
        self.teacher_method = None
        self.student_model = None
        self.teacher_model = None

        super().__init__(**args)

        self.student_method_name = self._normalize_method_name(
            self.hparams.get('student_method', None)
        )
        self.teacher_method_name = self._normalize_method_name(
            self.hparams.get('teacher_method', None)
        )

        if self.student_method_name is None:
            raise ValueError("`student_method` must be provided for KDMethod.")
        if self.teacher_method_name is None:
            raise ValueError("`teacher_method` must be provided for KDMethod.")

        from openstl.methods_vanilla import method_maps
        if self.student_method_name not in method_maps:
            raise ValueError(f"Unsupported student_method: {self.student_method_name}")
        if self.teacher_method_name not in method_maps:
            raise ValueError(f"Unsupported teacher_method: {self.teacher_method_name}")

        self.kd_weight = float(self.hparams.get('kd_weight', 0.5))

        # teacher is frozen
        for p in self.teacher_model.parameters():
            p.requires_grad = False
        self.teacher_model.eval()

        teacher_ckpt = self.hparams.get('teacher_ckpt', None)
        if teacher_ckpt is not None:
            self._load_teacher_ckpt(teacher_ckpt)

    # ------------------------------------------------------------------
    # Build wrappers
    # ------------------------------------------------------------------
    def _build_model(self, **args):
        """Build student method + teacher method, and return student model."""
        from openstl.methods_vanilla import method_maps
        from openstl.utils.main_utils import load_config

        student_method = self._normalize_method_name(args.get('student_method', None))
        teacher_method = self._normalize_method_name(args.get('teacher_method', None))

        if student_method is None:
            raise ValueError("`student_method` must be provided in args for KDMethod.")
        if teacher_method is None:
            raise ValueError("`teacher_method` must be provided in args for KDMethod.")

        if student_method not in method_maps:
            raise ValueError(f"Unsupported student_method: {student_method}")
        if teacher_method not in method_maps:
            raise ValueError(f"Unsupported teacher_method: {teacher_method}")

        # -------------------------
        # build student method
        # -------------------------
        student_args = dict(args)
        student_cfg_path = args.get('student_config_file', None)
        if student_cfg_path is not None:
            student_cfg = load_config(student_cfg_path)
            student_args.update(student_cfg)
        student_args['method'] = student_method

        self.student_method = method_maps[student_method](**student_args)
        self.student_model = self.student_method.model

        # -------------------------
        # build teacher method
        # -------------------------
        teacher_args = dict(args)
        teacher_cfg_path = args.get('teacher_config_file', None)
        if teacher_cfg_path is None:
            raise ValueError("`teacher_config_file` must be provided for KD teacher construction.")

        teacher_cfg = load_config(teacher_cfg_path)
        teacher_args.update(teacher_cfg)
        teacher_args['method'] = teacher_method

        self.teacher_method = method_maps[teacher_method](**teacher_args)
        self.teacher_model = self.teacher_method.model

        # return student model for optimizer / checkpoint / FLOPs logic
        return self.student_model

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _normalize_method_name(self, name):
        if name is None:
            return None
        return str(name).lower()

    def _load_teacher_ckpt(self, ckpt_path):
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Teacher checkpoint not found: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location='cpu')

        if ckpt_path.endswith('.ckpt'):
            state_dict = ckpt.get('state_dict', ckpt)
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('model.'):
                    new_state_dict[k[len('model.'):]] = v
                else:
                    new_state_dict[k] = v
            missing, unexpected = self.teacher_model.load_state_dict(new_state_dict, strict=False)
        else:
            missing, unexpected = self.teacher_model.load_state_dict(ckpt, strict=False)

        if len(missing) > 0:
            print(f"[KDMethod] Warning: missing teacher keys: {missing}")
        if len(unexpected) > 0:
            print(f"[KDMethod] Warning: unexpected teacher keys: {unexpected}")

    def _sync_runtime_states(self):
        """Sync outer KDMethod runtime states to inner method wrappers.

        This is critical because student_method / teacher_method are nested
        LightningModules not directly managed by Trainer.
        """
        if self.student_method is not None:
            self.student_method._kd_current_epoch = self.current_epoch
            self.student_method._kd_global_step = self.global_step

        if self.teacher_method is not None:
            self.teacher_method._kd_current_epoch = self.current_epoch
            self.teacher_method._kd_global_step = self.global_step

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------
    def forward(self, batch_x, batch_y=None, **kwargs):
        self._sync_runtime_states()
        return self.student_method.predict(batch_x, batch_y, **kwargs)

    def predict(self, batch_x, batch_y=None, **kwargs):
        self._sync_runtime_states()
        return self.student_method.predict(batch_x, batch_y, **kwargs)

    def compute_loss(self, batch, **kwargs):
        batch_x, batch_y = batch

        # Sync outer runtime states before calling nested methods
        self._sync_runtime_states()

        # student native training loss + prediction
        student_native_loss, pred_s = self.student_method.compute_loss(batch)

        # teacher prediction only
        with torch.no_grad():
            pred_t = self.teacher_method.predict(batch_x, batch_y)

        kd_loss = self.criterion(pred_s, pred_t)
        total_loss = student_native_loss + self.kd_weight * kd_loss

        extra = {
            'student_native_loss': student_native_loss.detach(),
            'kd_loss': kd_loss.detach(),
            'pred_s': pred_s,
            'pred_t': pred_t
        }
        return total_loss, pred_s, extra

    def training_step(self, batch, batch_idx):
        total_loss, _, extra = self.compute_loss(batch)

        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('student_native_loss', extra['student_native_loss'], on_step=True, on_epoch=True, prog_bar=False)
        self.log('kd_loss', extra['kd_loss'], on_step=True, on_epoch=True, prog_bar=False)

        return total_loss

    def validation_step(self, batch, batch_idx):
        self._sync_runtime_states()
        batch_x, batch_y = batch
        pred_y = self.student_method.predict(batch_x, batch_y)
        loss = self.criterion(pred_y, batch_y)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=False)
        return loss

    def test_step(self, batch, batch_idx):
        self._sync_runtime_states()
        batch_x, batch_y = batch
        pred_y = self.student_method.predict(batch_x, batch_y)
        outputs = {
            'inputs': batch_x.detach().cpu().numpy(),
            'preds': pred_y.detach().cpu().numpy(),
            'trues': batch_y.detach().cpu().numpy()
        }
        self.test_outputs.append(outputs)
        return outputs

    def on_train_start(self):
        self.teacher_model.eval()
        self._sync_runtime_states()

    def on_validation_start(self):
        self.teacher_model.eval()
        self._sync_runtime_states()

    def on_test_start(self):
        self.teacher_model.eval()
        self._sync_runtime_states()