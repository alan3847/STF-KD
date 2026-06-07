import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from openstl.methods_vanilla.base_method import Base_method


class _FeatureHook:
    def __init__(self, root_module, layer_name, output_index=0, keep_last_only=False):
        self.features = []
        self.output_index = output_index
        self.keep_last_only = keep_last_only
        self.enabled = False
        self.module = self._get_module(root_module, layer_name)
        self.handle = self.module.register_forward_hook(self._hook)

    def _get_module(self, root_module, layer_name):
        module = root_module
        for name in layer_name.split('.'):
            module = getattr(module, name)
        return module

    def _pick_tensor(self, output):
        if isinstance(output, (tuple, list)):
            return output[self.output_index]
        return output

    def _hook(self, module, inputs, output):
        if not self.enabled:
            return

        feat = self._pick_tensor(output)

        if self.keep_last_only:
            self.features = [feat]
        else:
            self.features.append(feat)

    def enable(self, clear=True):
        if clear:
            self.clear()
        self.enabled = True

    def disable(self, clear=False):
        self.enabled = False
        if clear:
            self.clear()

    def clear(self):
        self.features.clear()

    def close(self):
        self.handle.remove()

    def get(self, reduce='last'):
        if len(self.features) == 0:
            raise RuntimeError('No feature was captured by the hook.')

        if reduce == 'last':
            return self.features[-1]

        if reduce == 'mean':
            shapes = [tuple(f.shape) for f in self.features]
            if len(set(shapes)) != 1:
                return self.features[-1]
            return torch.stack(self.features, dim=0).mean(dim=0)

        raise ValueError(f'Unsupported feature reduce mode: {reduce}')


class FitNetKDMethod(Base_method):
    r"""FitNets hidden-layer distillation for OpenSTL.

    total_loss = student_native_loss
               + kd_weight * MSE(student_pred, teacher_pred)
               + hint_weight * MSE(regressor(student_hint), teacher_hint)
    """

    def __init__(self, **args):
        self.student_method = None
        self.teacher_method = None
        self.student_model = None
        self.teacher_model = None
        self.teacher_hook = None
        self.student_hook = None

        super().__init__(**args)

        self.student_method_name = self._normalize_method_name(
            self.hparams.get('student_method', None)
        )
        self.teacher_method_name = self._normalize_method_name(
            self.hparams.get('teacher_method', None)
        )

        self.kd_weight = float(self.hparams.get('kd_weight', 0.5))
        self.hint_weight = float(self.hparams.get('hint_weight', 1.0))
        self.hint_reduce = self.hparams.get('hint_reduce', 'last')

        teacher_hint_layer = self.hparams.get('teacher_hint_layer', None)
        student_hint_layer = self.hparams.get('student_hint_layer', None)
        if teacher_hint_layer is None or student_hint_layer is None:
            raise ValueError('FitNetKDMethod requires teacher_hint_layer and student_hint_layer.')

        teacher_index = int(self.hparams.get('teacher_hint_output_index', 0))
        student_index = int(self.hparams.get('student_hint_output_index', 0))

        self.teacher_hint_layer = teacher_hint_layer
        self.student_hint_layer = student_hint_layer
        self.teacher_hint_output_index = teacher_index
        self.student_hint_output_index = student_index

        self._get_module_by_path(self.teacher_method, self.teacher_hint_layer)
        self._get_module_by_path(self.student_method, self.student_hint_layer)

        keep_last_only = self.hint_reduce == 'last'

        self.teacher_hook = _FeatureHook(
            self.teacher_method,
            self.teacher_hint_layer,
            self.teacher_hint_output_index,
            keep_last_only=keep_last_only,
        )

        self.student_hook = _FeatureHook(
            self.student_method,
            self.student_hint_layer,
            self.student_hint_output_index,
            keep_last_only=keep_last_only,
        )


        self.hint_adapter = self._build_hint_adapter()

        for p in self.teacher_model.parameters():
            p.requires_grad = False
        self.teacher_method.eval()
        self.teacher_model.eval()

        teacher_ckpt = self.hparams.get('teacher_ckpt', None)
        if teacher_ckpt is not None:
            self._load_model_ckpt(self.teacher_model, teacher_ckpt, 'teacher')

        student_ckpt = self.hparams.get('student_ckpt', None)
        if student_ckpt is not None:
            self._load_model_ckpt(self.student_model, student_ckpt, 'student')

    def _build_model(self, **args):
        from openstl.methods_vanilla import method_maps
        from openstl.utils.main_utils import load_config

        student_method = self._normalize_method_name(args.get('student_method', None))
        teacher_method = self._normalize_method_name(args.get('teacher_method', None))

        if student_method not in method_maps:
            raise ValueError(f'Unsupported student_method: {student_method}')
        if teacher_method not in method_maps:
            raise ValueError(f'Unsupported teacher_method: {teacher_method}')

        student_args = dict(args)
        student_cfg_path = args.get('student_config_file', None)
        if student_cfg_path is not None:
            student_args.update(load_config(student_cfg_path))
        student_args['method'] = student_method
        self.student_method = method_maps[student_method](**student_args)
        self.student_model = self.student_method.model

        teacher_args = dict(args)
        teacher_cfg_path = args.get('teacher_config_file', None)
        if teacher_cfg_path is None:
            raise ValueError('teacher_config_file must be provided for FitNet teacher.')
        teacher_args.update(load_config(teacher_cfg_path))
        teacher_args['method'] = teacher_method
        self.teacher_method = method_maps[teacher_method](**teacher_args)
        self.teacher_model = self.teacher_method.model

        return self.student_model

    def _build_hint_adapter(self):
        teacher_channels = self.hparams.get('hint_teacher_channels', None)
        student_channels = self.hparams.get('hint_student_channels', None)

        if teacher_channels is None:
            return nn.Identity()

        teacher_channels = int(teacher_channels)
        if student_channels is None:
            return nn.LazyConv2d(teacher_channels, kernel_size=1, bias=True)

        student_channels = int(student_channels)
        return nn.Conv2d(student_channels, teacher_channels, kernel_size=1, bias=True)

    def configure_optimizers(self):
        from openstl.core import get_optim_scheduler

        optim_model = nn.ModuleList([self.student_model, self.hint_adapter])
        optimizer, scheduler, by_epoch = get_optim_scheduler(
            self.hparams,
            self.hparams.epoch,
            optim_model,
            self.hparams.steps_per_epoch
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch' if by_epoch else 'step'
            },
        }

    def _normalize_method_name(self, name):
        return None if name is None else str(name).lower()

    def _get_module_by_path(self, root, path):
        obj = root
        for part in path.split('.'):
            if part.isdigit():
                obj = obj[int(part)]
            else:
                obj = getattr(obj, part)
        return obj

    def _load_model_ckpt(self, model, ckpt_path, role):
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f'{role} checkpoint not found: {ckpt_path}')

        ckpt = torch.load(ckpt_path, map_location='cpu')
        state_dict = ckpt.get('state_dict', ckpt) if isinstance(ckpt, dict) else ckpt

        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('model.'):
                new_state_dict[k[len('model.'):]] = v
            else:
                new_state_dict[k] = v

        missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
        if len(missing) > 0:
            print(f'[FitNetKDMethod] Warning: missing {role} keys: {missing}')
        if len(unexpected) > 0:
            print(f'[FitNetKDMethod] Warning: unexpected {role} keys: {unexpected}')

    def _sync_runtime_states(self):
        if self.student_method is not None:
            self.student_method._kd_current_epoch = self.current_epoch
            self.student_method._kd_global_step = self.global_step
        if self.teacher_method is not None:
            self.teacher_method._kd_current_epoch = self.current_epoch
            self.teacher_method._kd_global_step = self.global_step

    def _to_nchw(self, feat, name):
        if feat.dim() == 5:
            b, t, c, h, w = feat.shape
            return feat.reshape(b * t, c, h, w)
        if feat.dim() == 4:
            return feat
        if feat.dim() == 2:
            return feat[:, :, None, None]
        raise ValueError(f'{name} hint feature must be 2D/4D/5D, got shape {tuple(feat.shape)}')

    def _align_batch(self, student_feat, teacher_feat):
        bs, bt = student_feat.shape[0], teacher_feat.shape[0]
        if bs == bt:
            return student_feat, teacher_feat

        if bs > bt and bs % bt == 0:
            student_feat = student_feat.reshape(bt, bs // bt, *student_feat.shape[1:]).mean(dim=1)
        elif bt > bs and bt % bs == 0:
            teacher_feat = teacher_feat.reshape(bs, bt // bs, *teacher_feat.shape[1:]).mean(dim=1)
        else:
            raise ValueError(
                f'Cannot align hint batch sizes: student={bs}, teacher={bt}. '
                'Choose layers with compatible temporal/batch dimensions.'
            )
        return student_feat, teacher_feat

    def _compute_hint_loss(self, student_feat, teacher_feat):
        student_feat = self._to_nchw(student_feat, 'student')
        teacher_feat = self._to_nchw(teacher_feat, 'teacher').detach()

        student_feat, teacher_feat = self._align_batch(student_feat, teacher_feat)

        student_feat = self.hint_adapter(student_feat)

        if student_feat.shape[-2:] != teacher_feat.shape[-2:]:
            student_feat = F.interpolate(
                student_feat,
                size=teacher_feat.shape[-2:],
                mode='bilinear',
                align_corners=False
            )

        if student_feat.shape[1] != teacher_feat.shape[1]:
            raise ValueError(
                f'Hint channel mismatch after adapter: student={student_feat.shape[1]}, '
                f'teacher={teacher_feat.shape[1]}. Set --hint_teacher_channels correctly.'
            )

        return F.mse_loss(student_feat, teacher_feat)

    def forward(self, batch_x, batch_y=None, **kwargs):
        self._sync_runtime_states()
        return self.student_method.predict(batch_x, batch_y, **kwargs)

    def predict(self, batch_x, batch_y=None, **kwargs):
        return self.forward(batch_x, batch_y, **kwargs)

    def compute_loss(self, batch, **kwargs):
        batch_x, batch_y = batch
        self._sync_runtime_states()

        self.student_hook.clear()
        student_native_loss, pred_s = self.student_method.compute_loss(batch)
        student_feat = self.student_hook.get(self.hint_reduce)
        if student_feat is None:
            raise RuntimeError('Student hint layer did not run. Check student_hint_layer.')

        self.teacher_hook.clear()
        self.teacher_method.eval()
        self.teacher_model.eval()
        with torch.no_grad():
            pred_t = self.teacher_method.predict(batch_x, batch_y)
        teacher_feat = self.teacher_hook.get(self.hint_reduce)
        if teacher_feat is None:
            raise RuntimeError('Teacher hint layer did not run. Check teacher_hint_layer.')

        kd_loss = self.criterion(pred_s, pred_t.detach())
        hint_loss = self._compute_hint_loss(student_feat, teacher_feat)

        total_loss = student_native_loss + self.kd_weight * kd_loss + self.hint_weight * hint_loss

        extra = {
            'student_native_loss': student_native_loss.detach(),
            'kd_loss': kd_loss.detach(),
            'hint_loss': hint_loss.detach(),
            'pred_s': pred_s,
            'pred_t': pred_t,
        }
        return total_loss, pred_s, extra

    def training_step(self, batch, batch_idx):
        total_loss, _, extra = self.compute_loss(batch)

        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('student_native_loss', extra['student_native_loss'], on_step=True, on_epoch=True)
        self.log('kd_loss', extra['kd_loss'], on_step=True, on_epoch=True)
        self.log('hint_loss', extra['hint_loss'], on_step=True, on_epoch=True)

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
        self.teacher_method.eval()
        self.teacher_model.eval()
        self._sync_runtime_states()

    def on_validation_start(self):
        self.teacher_method.eval()
        self.teacher_model.eval()
        self._sync_runtime_states()

    def on_test_start(self):
        self.teacher_method.eval()
        self.teacher_model.eval()
        self._sync_runtime_states()

    def teardown(self, stage=None):
        if self.teacher_hook is not None:
            self.teacher_hook.close()
        if self.student_hook is not None:
            self.student_hook.close()

