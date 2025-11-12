"""Portions adapted from Shape As Points (https://github.com/autonomousvision/shape_as_points, MIT); see THIRD_PARTY_NOTICES.md.

Shape-As-Points style inverse rendering step (Peng et al. 2021), adapted to the MPU particle pipeline.

Oriented sample particles -> differentiable Poisson (DPSR) indicator grid -> marching cubes with the SAP
surrogate gradient (dV/dchi = -n, splatted back onto the grid) -> nvdiffrast -> image losses.  The gradient
reaches *every* particle position and normal through the spectral Poisson solve, instead of only the
silhouette vertices of the MPU marching-cubes mesh.  Particles are updated with Adam (SAP), then the MPU is
re-fitted so that the representation, resampling and evaluation stay identical to the rest of the pipeline.
"""
import time
import numpy as np
import torch
import torch.nn.functional as Fnn

from .dpsr import DPSR, point_rasterize
from .recon import run_marching_cubes


def _sample_grid_vec(g, u):
    """g (R,R,R,3) grid of vectors (x,y,z index order, node i at u=i/R); u (M,3) in [0,1) -> (M,3)."""
    R = g.shape[0]
    c = -1.0 + 2.0 * u * R / (R - 1)                      # align_corners=True normalised coordinate
    grid = torch.stack([c[:, 2], c[:, 1], c[:, 0]], -1)   # grid_sample wants (W,H,D) = (z,y,x) order
    vol = g.permute(3, 0, 1, 2)[None]                     # (1,3,X,Y,Z)
    out = Fnn.grid_sample(vol, grid[None, None, None], mode="bilinear", padding_mode="border", align_corners=True)
    return out[0, :, 0, 0, :].t()                         # (M,3)


class PSR2Mesh(torch.autograd.Function):
    """Marching cubes of an indicator grid chi (R,R,R) (negative inside) with the Shape-As-Points backward:
    dL/dchi = splat( -(dL/dV . n) ) at the vertex positions."""

    @staticmethod
    def forward(ctx, chi):
        R = chi.shape[0]
        V, F = run_marching_cubes(chi.detach().contiguous(), R, device=chi.device, corner_aligned=True)
        Vt = torch.as_tensor(np.ascontiguousarray(V), dtype=torch.float32, device=chi.device)
        Ft = torch.as_tensor(np.ascontiguousarray(F), dtype=torch.long, device=chi.device)
        if Vt.shape[0] == 0:
            ctx.save_for_backward(torch.zeros(0, 3, device=chi.device), torch.zeros(0, 3, device=chi.device))
            ctx.R = R
            return Vt, Ft, torch.zeros(0, 3, device=chi.device)
        with torch.no_grad():
            gx, gy, gz = torch.gradient(chi.detach(), spacing=2.0 / R)
            g = torch.stack([gx, gy, gz], -1)
            u = ((Vt + 1.0) * 0.5).clamp(0.0, 1.0 - 1e-6)
            n = _sample_grid_vec(g, u)
            n = n / n.norm(dim=1, keepdim=True).clamp_min(1e-8)   # outward (chi negative inside)
        ctx.save_for_backward(u, n)
        ctx.R = R
        return Vt, Ft, n

    @staticmethod
    def backward(ctx, gV, gF, gN):
        u, n = ctx.saved_tensors
        R = ctx.R
        if u.shape[0] == 0:
            return torch.zeros((R, R, R), device=u.device)
        s = -(gV * n).sum(1, keepdim=True)                   # dV/dchi = -n  (SAP)
        grad_chi = point_rasterize(u[None].float(), s[None].float(), (R, R, R))[0, 0]
        return grad_chi


class SapState:
    def __init__(self):
        self.t = 0
        self.m_p = self.v_p = self.m_n = self.v_n = None
        self.n = -1

    def reset(self, n, device):
        self.t = 0
        self.n = n
        self.m_p = torch.zeros(n, 3, device=device); self.v_p = torch.zeros(n, 3, device=device)
        self.m_n = torch.zeros(n, 3, device=device); self.v_n = torch.zeros(n, 3, device=device)


def _adam(m, v, g, t, lr, b1=0.9, b2=0.999, eps=1e-8):
    m.mul_(b1).add_(g, alpha=1 - b1)
    v.mul_(b2).addcmul_(g, g, value=1 - b2)
    mhat = m / (1 - b1 ** t)
    vhat = v / (1 - b2 ** t)
    return lr * mhat / (vhat.sqrt() + eps)


def step_sap(D, it):
    """One SAP-style optimisation step on DiffMPU D. Returns the usual info dict."""
    from .render_utils import img_loss
    a = D.args
    t0 = time.time()
    if not hasattr(D, "dpsr"):
        D.dpsr = DPSR((a.dpsr_res,) * 3, sig=a.dpsr_sig).to(D.dev)
    if not hasattr(D, "sap"):
        D.sap = SapState()
    n = D.sp_pos.shape[0]
    if D.sap.n != n:
        D.sap.reset(n, D.dev)
    p01 = ((D.sp_pos.detach() + 1.0) * 0.5).clamp(1e-6, 1 - 1e-6).requires_grad_(True)
    nrm = D.sp_nrm.detach().clone().requires_grad_(True)
    nn_ = nrm / nrm.norm(dim=1, keepdim=True).clamp_min(1e-8)
    chi = torch.tanh(D.dpsr(p01[None], nn_[None])[0])
    V, F, VN = PSR2Mesh.apply(chi)
    if V.shape[0] == 0:
        return {"it": it, "loss": float("nan"), "psnr": None, "n_sp": n, "n_fp": D.R.mpu.fp_stats()["n_fp"],
                "n_del": 0, "mean_step": 0.0, "t_vel": time.time() - t0, "t_build": 0.0}
    imgs = D.renderer.render(V, F, VN)
    photo_w = a.photo_w if it >= a.photo_start else 0.0
    loss = photo_w * img_loss(imgs, D.target, D.renderer.kernel, multi_scale=True)
    if a.mask_w > 0:
        masks = D.renderer.render_mask(V, F)
        loss = loss + a.mask_w * img_loss(masks, D.target_mask, D.renderer.kernel, multi_scale=True)
    loss.backward()
    with torch.no_grad():
        ps = float(-10 * torch.log10((imgs - D.target).square().mean().clamp_min(1e-12)))
        gp = torch.nan_to_num(p01.grad); gn = torch.nan_to_num(nrm.grad)
        D.sap.t += 1
        lr = a.sap_lr if it < a.lr_switch else a.sap_lr2
        dp01 = _adam(D.sap.m_p, D.sap.v_p, gp, D.sap.t, lr)
        dn = _adam(D.sap.m_n, D.sap.v_n, gn, D.sap.t, lr * a.sap_nlr_scale)
        # cap the per-step motion like the MPU flow (in world units)
        dp = -2.0 * dp01
        max_step = (a.max_step_cells if it < a.lr_switch else a.max_step_cells2) * D.cell_fine
        nrm_dp = dp.norm(dim=1, keepdim=True)
        dp = torch.where(nrm_dp > max_step, dp * (max_step / nrm_dp.clamp_min(1e-12)), dp)
        newp = (D.sp_pos + dp).clamp(-0.999, 0.999)
        newn = nrm - dn
        newn = newn / newn.norm(dim=1, keepdim=True).clamp_min(1e-8)
    if a.hull_w > 0 and it >= a.hull_start:
        hd, _ = D.hull_displacement(newp)          # uses autograd internally -> outside no_grad
        with torch.no_grad():
            hn = hd.norm(dim=1, keepdim=True)
            hmax = a.hull_max_cells * D.cell_fine
            hd = torch.where(hn > hmax, hd * (hmax / hn.clamp_min(1e-12)), hd)
            newp = (newp + a.hull_w * hd).clamp(-0.999, 0.999)
    with torch.no_grad():
        D.sp_pos, D.sp_nrm, D.sp_vel = newp.detach(), newn.detach(), dp
    t_vel = time.time() - t0
    t1 = time.time()
    D.build()
    t_build = time.time() - t1
    D.it = it
    info = {"it": it, "loss": float(loss.item()), "psnr": ps, "n_sp": int(D.sp_pos.shape[0]), "n_fp": D.R.mpu.fp_stats()["n_fp"],
            "n_del": 0, "mean_step": float(dp.norm(dim=1).mean()), "t_vel": t_vel, "t_build": t_build}
    if a.resample_every > 0 and it % a.resample_every == 0 and it > 0:
        t2 = time.time()
        mode = a.resample_mode
        if getattr(a, "resample_mode2", "same") != "same" and it >= a.lr_switch:
            mode = a.resample_mode2
        if mode != "none":
            D.resample(mode)
            D.sap.reset(D.sp_pos.shape[0], D.dev)
        info["t_resample"] = time.time() - t2
    D.log.append(info)
    return info
