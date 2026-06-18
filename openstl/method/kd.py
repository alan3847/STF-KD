import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from openstl.methods.base_method import BaseMethod
from openstl.core import metric

class GRAHSC_KD(BaseMethod):
    def __init__(self, args, device):
        super(GRAHSC_KD, self).__init__(args, device)
        self.beta_min = 0.1
        self.beta_max = 0.9
        self.gamma = 1.0
        self.tau = 0.05
        self.alpha1 = 0.3
        self.alpha2 = 0.8
        self.zeta_min = 0.2
        self.zeta_max = 0.7
        self.total_iters = args.max_iterations
        self.omega_a = None
        self.omega_b = None

    def _init_affine_params(self, student_pred):
        B, T_out, H, W, C = student_pred.shape
        self.omega_a = nn.Parameter(torch.ones_like(student_pred), requires_grad=True).to(self.device)
        self.omega_b = nn.Parameter(torch.zeros_like(student_pred), requires_grad=True).to(self.device)

    def global_affine_proj(self, y_s):
        return self.omega_a * y_s + self.omega_b

    def compute_gra_loss(self, y_t, y_s, cur_iter):
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
        B, T_out, H, C = y_s.shape
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

    def training_one_step(self, batch_data, teacher_model, student_model, cur_iter):
        x, y = batch_data
        x = x.to(self.device)
        y = y.to(self.device)
        with torch.no_grad():
            y_teacher = teacher_model(x)
        y_student = student_model(x)
        L_pred = torch.norm(y_student - y, p='fro') ** 2 / y_student.numel()
        _, _, L_GRA = self.compute_gra_loss(y_teacher, y_student, cur_iter)
        L_HSC = self.compute_hsc_loss(y_teacher, y_student, cur_iter)
        L_KD = L_GRA + L_HSC
        total_loss = L_pred + L_KD
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        log_dict = {}
        log_dict["train/total_loss"] = total_loss.item()
        log_dict["train/L_pred"] = L_pred.item()
        log_dict["train/L_GRA"] = L_GRA.item()
        log_dict["train/L_HSC"] = L_HSC.item()
        mse = metric.mse(y_student, y)
        log_dict["train/mse"] = mse
        return total_loss, log_dict

    def validation_one_step(self, batch_data, teacher_model, student_model):
        x, y = batch_data
        x, y = x.to(self.device), y.to(self.device)
        with torch.no_grad():
            y_t = teacher_model(x)
            y_s = student_model(x)
            L_pred = torch.norm(y_s - y, p='fro') ** 2 / y_s.numel()
            _, _, L_GRA = self.compute_gra_loss(y_t, y_s, 0)
            L_HSC = self.compute_hsc_loss(y_t, y_s, 0)
            total_loss = L_pred + L_GRA + L_HSC
            mse = metric.mse(y_s, y)
        log_dict = {}
        log_dict["val/total_loss"] = total_loss.item()
        log_dict["val/mse"] = mse
        return log_dict

    def configure_optimizers(self, model):
        params = list(model.parameters())
        if self.omega_a is not None:
            params += [self.omega_a, self.omega_b]
        optimizer = Adam(params, lr=self.args.lr)
        return optimizer