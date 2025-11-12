"""Portions adapted from Shape As Points (https://github.com/autonomousvision/shape_as_points, MIT); see THIRD_PARTY_NOTICES.md.

Differentiable Poisson Surface Reconstruction (Peng et al. 2021, Shape As Points), compact port.
Points must lie in [0,1)^3. Returns an indicator grid (negative inside after the sign convention below)."""
import numpy as np
import torch


def fftfreqs(res, dtype=torch.float32):
    freqs = []
    for dim in range(len(res) - 1):
        r_ = res[dim]
        freqs.append(torch.tensor(np.fft.fftfreq(r_, d=1 / r_), dtype=dtype))
    r_ = res[-1]
    freqs.append(torch.tensor(np.fft.rfftfreq(r_, d=1 / r_), dtype=dtype))
    omega = torch.meshgrid(*freqs, indexing="ij")
    return torch.stack(list(omega), dim=-1)


def img(x, deg=1):
    deg %= 4
    if deg == 0:
        return x
    if deg == 1:
        res = x[..., [1, 0]].clone()
        res[..., 0] = -res[..., 0]
        return res
    if deg == 2:
        return -x
    res = x[..., [1, 0]].clone()
    res[..., 1] = -res[..., 1]
    return res


def spec_gaussian_filter(res, sig):
    omega = fftfreqs(res, dtype=torch.float64)
    dis = torch.sqrt(torch.sum(omega ** 2, dim=-1))
    return torch.exp(-0.5 * ((sig * 2 * dis / res[0]) ** 2)).unsqueeze(-1).unsqueeze(-1)


def point_rasterize(pts, vals, size):
    """Trilinear splat of vals at pts (in [0,1)) onto a grid. pts (B,N,3), vals (B,N,F) -> (B,F,*size)."""
    B, N, dim = pts.shape
    nf = vals.shape[-1]
    dev = pts.device
    size_t = torch.tensor(size, device=dev, dtype=pts.dtype)
    cube = 1.0 / size_t
    ind0 = torch.floor(pts / cube).long()
    ind1 = torch.fmod(torch.ceil(pts / cube), size_t).long()
    out = torch.zeros((B, nf) + tuple(size), device=dev, dtype=vals.dtype)
    x0 = ind0.type(pts.dtype) * cube
    frac = (pts - x0) / cube                      # (B,N,3) in [0,1)
    for corner in range(8):
        bits = [(corner >> d) & 1 for d in range(3)]
        idx = torch.stack([ind1[..., d] if bits[d] else ind0[..., d] for d in range(3)], dim=-1)
        w = torch.ones_like(frac[..., 0])
        for d in range(3):
            w = w * (frac[..., d] if bits[d] else (1.0 - frac[..., d]))
        flat = (idx[..., 0] * size[1] + idx[..., 1]) * size[2] + idx[..., 2]        # (B,N)
        for b in range(B):
            for f in range(nf):
                out[b, f].view(-1).index_add_(0, flat[b], w[b] * vals[b, :, f])
    return out


def grid_interp(grid, pts):
    """grid (B,*size,1), pts (B,N,3) in [0,1) -> (B,N)"""
    B, N, _ = pts.shape
    size = torch.tensor(grid.shape[1:-1], device=grid.device, dtype=pts.dtype)
    cube = 1.0 / size
    ind0 = torch.floor(pts / cube).long()
    ind1 = torch.fmod(torch.ceil(pts / cube), size).long()
    frac = (pts - ind0.type(pts.dtype) * cube) / cube
    out = torch.zeros(B, N, device=grid.device, dtype=grid.dtype)
    for corner in range(8):
        bits = [(corner >> d) & 1 for d in range(3)]
        idx = torch.stack([ind1[..., d] if bits[d] else ind0[..., d] for d in range(3)], dim=-1)
        w = torch.ones_like(frac[..., 0])
        for d in range(3):
            w = w * (frac[..., d] if bits[d] else (1.0 - frac[..., d]))
        for b in range(B):
            out[b] += w[b] * grid[b, idx[b, :, 0], idx[b, :, 1], idx[b, :, 2], 0]
    return out


class DPSR(torch.nn.Module):
    def __init__(self, res, sig=2.0, scale=True, shift=True):
        super().__init__()
        self.res = tuple(res)
        self.sig = sig
        self.dim = len(res)
        self.register_buffer("G", spec_gaussian_filter(res=self.res, sig=sig).float())
        self.omega = fftfreqs(self.res, dtype=torch.float32).unsqueeze(-1) * (2 * np.pi)
        self.scale = scale
        self.shift = shift

    def forward(self, V, N):
        """V, N: (B, n, 3) with V in [0,1). Returns phi (B, *res): negative inside."""
        ras_p = point_rasterize(V, N, self.res)                     # (B,3,r,r,r)
        ras_s = torch.fft.rfftn(ras_p, dim=(2, 3, 4))               # (B,3,r,r,r/2+1)
        ras_s = ras_s.permute(0, 2, 3, 4, 1)                        # (B,r,r,r/2+1,3)
        N_ = ras_s[..., None] * self.G.to(V.device)                 # complex (B,r,r,r/2+1,3,1)
        omega = self.omega.to(V.device)                              # (r,r,r/2+1,3,1)
        DivN = torch.sum(-img(torch.view_as_real(N_[..., 0])) * omega, dim=-2)   # (B,r,r,r/2+1,2)
        Lap = -torch.sum(omega ** 2, -2)                             # (r,r,r/2+1,1)
        Phi = DivN / (Lap + 1e-6)
        Phi = Phi.permute(1, 2, 3, 4, 0)
        Phi[0, 0, 0] = 0
        Phi = Phi.permute(4, 0, 1, 2, 3)
        phi = torch.fft.irfftn(torch.view_as_complex(Phi.contiguous()), s=self.res, dim=(1, 2, 3))
        if self.shift or self.scale:
            fv = grid_interp(phi.unsqueeze(-1), V)
            if self.shift:
                phi = phi - fv.mean(dim=-1).view(-1, 1, 1, 1)
            if self.scale:
                fv0 = phi[:, 0, 0, 0]
                phi = -phi / fv0.abs().view(-1, 1, 1, 1) * 0.5
        return phi
