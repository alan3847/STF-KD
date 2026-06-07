import torch
import torch.nn.functional as F

from openstl.methods_fitnet.fitnet_kd_method import FitNetKDMethod


class FrequencyAlignedKDMethod(FitNetKDMethod):
    r"""Frequency-aligned latent KD.

    total_loss = student_native_loss
               + kd_weight * MSE(student_pred, teacher_pred)
               + fakd_weight * (high_freq_loss + freq_alpha * low_freq_loss)

    This is an engineering version of frequency-aligned distillation:
    teacher/student latent features are split into low/high frequency bands
    by FFT masks, instead of requiring an explicitly frequency-decoupled teacher.
    """

    def __init__(self, **args):
        super().__init__(**args)

        self.fakd_weight = float(self.hparams.get('fakd_weight', self.hparams.get('hint_weight', 1.0)))
        self.freq_alpha = float(self.hparams.get('freq_alpha', 1.0))
        self.freq_cutoff = float(self.hparams.get('freq_cutoff', 0.25))
        self.freq_loss_type = self.hparams.get('freq_loss_type', 'complex')
        self.output_kd = bool(self.hparams.get('output_kd', True))

        self.freq_norm = self.hparams.get('freq_norm', 'rms')
        self.freq_eps = float(self.hparams.get('freq_eps', 1e-6))
        self.freq_log_mag = bool(self.hparams.get('freq_log_mag', False))

        if self.freq_loss_type not in ['complex', 'magnitude']:
            raise ValueError(f'Unsupported freq_loss_type: {self.freq_loss_type}')

        if self.freq_norm not in ['none', 'rms', 'standard']:
            raise ValueError(f'Unsupported freq_norm: {self.freq_norm}')

    def _normalize_feature_for_fft(self, feat):
        if self.freq_norm == 'none':
            return feat

        if self.freq_norm == 'rms':
            denom = feat.pow(2).mean(dim=(-2, -1), keepdim=True).sqrt()
            return feat / denom.clamp_min(self.freq_eps)

        if self.freq_norm == 'standard':
            mean = feat.mean(dim=(-2, -1), keepdim=True)
            centered = feat - mean
            std = centered.pow(2).mean(dim=(-2, -1), keepdim=True).sqrt()
            return centered / std.clamp_min(self.freq_eps)

        raise ValueError(f'Unsupported freq_norm: {self.freq_norm}')


    def _build_frequency_masks(self, h, w, device, dtype):
        cutoff = max(0.0, min(float(self.freq_cutoff), 0.5))

        fy = torch.fft.fftfreq(h, d=1.0, device=device)
        fx = torch.fft.fftfreq(w, d=1.0, device=device)
        radius = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)

        low_mask = (radius <= cutoff).to(dtype=dtype)
        high_mask = 1.0 - low_mask
        return low_mask[None, None], high_mask[None, None]


    def _masked_freq_loss(self, student_freq, teacher_freq, mask):
        if self.freq_loss_type == 'magnitude':
            student_mag = student_freq.abs()
            teacher_mag = teacher_freq.abs()

            if self.freq_log_mag:
                student_mag = torch.log1p(student_mag)
                teacher_mag = torch.log1p(teacher_mag)

            diff = (student_mag - teacher_mag).pow(2)
        else:
            diff = (student_freq - teacher_freq).abs().pow(2)

        denom = mask.sum().clamp_min(1.0) * diff.shape[0] * diff.shape[1]
        return (diff * mask).sum() / denom


    def _compute_frequency_losses(self, student_feat, teacher_feat):
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
                f'Feature channel mismatch after adapter: student={student_feat.shape[1]}, '
                f'teacher={teacher_feat.shape[1]}. Set --hint_teacher_channels and '
                f'--hint_student_channels correctly.'
            )

        student_feat = student_feat.float()
        teacher_feat = teacher_feat.float()

        student_feat = self._normalize_feature_for_fft(student_feat)
        teacher_feat = self._normalize_feature_for_fft(teacher_feat)

        student_freq = torch.fft.fft2(student_feat, dim=(-2, -1), norm='ortho')
        teacher_freq = torch.fft.fft2(teacher_feat, dim=(-2, -1), norm='ortho')


        _, _, h, w = student_feat.shape
        low_mask, high_mask = self._build_frequency_masks(
            h, w, student_feat.device, student_feat.dtype
        )

        low_loss = self._masked_freq_loss(student_freq, teacher_freq, low_mask)
        high_loss = self._masked_freq_loss(student_freq, teacher_freq, high_mask)
        freq_loss = high_loss + self.freq_alpha * low_loss

        return freq_loss, high_loss, low_loss

    def compute_loss(self, batch, **kwargs):
        batch_x, batch_y = batch
        self._sync_runtime_states()

        self.student_hook.enable(clear=True)
        try:
            student_native_loss, pred_s = self.student_method.compute_loss(batch)
        finally:
            self.student_hook.disable()

        student_feat = self.student_hook.get(self.hint_reduce)

        if student_feat is None:
            raise RuntimeError('Student hint layer did not run. Check student_hint_layer.')

        self.teacher_hook.enable(clear=True)
        self.teacher_method.eval()
        self.teacher_model.eval()
        try:
            with torch.no_grad():
                pred_t = self.teacher_method.predict(batch_x, batch_y)
        finally:
            self.teacher_hook.disable()

        teacher_feat = self.teacher_hook.get(self.hint_reduce)

        if teacher_feat is None:
            raise RuntimeError('Teacher hint layer did not run. Check teacher_hint_layer.')

        if self.output_kd:
            kd_loss = self.criterion(pred_s, pred_t.detach())
        else:
            kd_loss = pred_s.new_zeros(())

        freq_loss, high_freq_loss, low_freq_loss = self._compute_frequency_losses(
            student_feat, teacher_feat
        )

        total_loss = (
            student_native_loss
            + self.kd_weight * kd_loss
            + self.fakd_weight * freq_loss
        )

        extra = {
            'student_native_loss': student_native_loss.detach(),
            'kd_loss': kd_loss.detach(),
            'freq_loss': freq_loss.detach(),
            'high_freq_loss': high_freq_loss.detach(),
            'low_freq_loss': low_freq_loss.detach(),
            'pred_s': pred_s,
            'pred_t': pred_t,
        }
        return total_loss, pred_s, extra

    def training_step(self, batch, batch_idx):
        total_loss, _, extra = self.compute_loss(batch)

        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('student_native_loss', extra['student_native_loss'], on_step=True, on_epoch=True)
        self.log('kd_loss', extra['kd_loss'], on_step=True, on_epoch=True)
        self.log('freq_loss', extra['freq_loss'], on_step=True, on_epoch=True)
        self.log('high_freq_loss', extra['high_freq_loss'], on_step=True, on_epoch=True)
        self.log('low_freq_loss', extra['low_freq_loss'], on_step=True, on_epoch=True)

        return total_loss
