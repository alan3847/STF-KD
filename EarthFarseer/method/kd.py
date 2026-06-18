import torch
import torch.nn as nn
from torch.optim import Adam

class GRAHSC_KD:
    def __init__(self, device, max_iter):
        self.device = device
        self.beta_min = 0.1
        self.beta_max = 0.9
        self.gamma = 1.0
        self.tau = 0.05
        self.alpha1 = 0.3
        self.alpha2 = 0.8
        self.zeta_min = 0.2
        self.zeta_max = 0.7
        self.total_iters = max_iter
        self.omega_a = None
        self.omega_b = None

    def _init_affine_params(self, student_pred):
        B, T_out, H, W, C = student_pred.shape
        self.omega_a = nn.Parameter(torch.ones_like(student_pred), requires_grad=True).to(self.device)
        self.omega_b = nn.Parameter(torch.zeros_like(student_pred), requires_grad=True).to(self.device)

    def global_affine_proj(self, y_s):
        return self.omega_a * y_s + self.omega_b

    def compute_gra_loss(self, y_t, y_s):
        if self.omega_a is None or self.omega_a.shape != y_s.shape:
            self._init_affine_params(y_s)
        y_s_align = self.global_affine_proj(y_s)
        delta_drift = y_s_align - y_t
        delta_res = (y_t - y_s) - (y_s_align - y_s)
        L_drift = torch.norm(delta_drift, p='fro') ** 2 / delta_drift.numel()
        L_res = torch.mean(torch.abs(delta_res))
        beta_t = self.beta_min + (self.beta_max - self.beta_min) * torch.sigmoid(self.gamma * L_drift)
        L_GRA = L_drift + beta_t * L_res
        return L_drift, L_res, L_GRA

    def split_hsc_region(self, y_s):
        var_map = torch.var(y_s, dim=0, keepdim=True)
        mask_abr = (var_map > self.tau).float()
        mask_sta = 1.0 - mask_abr
        return mask_sta, mask_abr

    def compute_hsc_loss(self, y_t, y_s, cur_iter):
        mask_sta, mask_abr = self.split_hsc_region(y_s)
        progress = torch.tensor(cur_iter / self.total_iters, dtype=torch.float32).to(self.device)
        zeta_t = self.zeta_min + (self.zeta_max - self.zeta_min) * torch.sigmoid(progress)
        w_sta = self.alpha1 * zeta_t
        w_abr = self.alpha2 * zeta_t
        base_mse = torch.square(y_s - y_t)
        loss_sta = torch.sum(w_sta * mask_sta * base_mse) / (mask_sta.sum() + 1e-8)
        loss_abr = torch.sum(w_abr * mask_abr * base_mse) / (mask_abr.sum() + 1e-8)
        L_HSC = loss_sta + loss_abr
        return L_HSC

    def calculate_total_loss(self, pred_student, pred_teacher, label, cur_iter):
        L_pred = torch.norm(pred_student - label, p='fro') ** 2 / pred_student.numel()
        _, _, L_GRA = self.compute_gra_loss(pred_teacher, pred_student)
        L_HSC = self.compute_hsc_loss(pred_teacher, pred_student, cur_iter)
        total_loss = L_pred + L_GRA + L_HSC
        return total_loss, L_pred, L_GRA, L_HSC

    def get_optimizer(self, model, lr):
        params = list(model.parameters())
        if self.omega_a is not None:
            params.extend([self.omega_a, self.omega_b])
        opt = Adam(params, lr=lr)
        return opt