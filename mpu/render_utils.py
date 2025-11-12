"""Portions adapted from Neural Implicit Evolution (https://github.com/ishit/nie, no licence declared upstream, used with attribution); see THIRD_PARTY_NOTICES.md.

Differentiable rendering utilities (nvdiffrast) and mesh operators, ported from NIE."""
import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation as Rot


def projection(x=0.1, n=1.0, f=50.0):
    return np.array([[n / x, 0, 0, 0],
                     [0, n / -x, 0, 0],
                     [0, 0, -(f + n) / (f - n), -(2 * f * n) / (f - n)],
                     [0, 0, -1, 0]]).astype(np.float32)


def translate(x, y, z):
    m = np.eye(4, dtype=np.float32)
    m[:3, 3] = [x, y, z]
    return m


def random_rotation(rng):
    r = Rot.random(random_state=rng).as_matrix()
    m = np.eye(4, dtype=np.float32)
    m[:3, :3] = r
    return m


def compute_edges(faces):
    v0, v1, v2 = faces.chunk(3, dim=1)
    edges = torch.cat([torch.cat([v1, v2], 1), torch.cat([v2, v0], 1), torch.cat([v0, v1], 1)], dim=0).long()
    edges, _ = edges.sort(dim=1)
    return torch.unique(edges, dim=0)


def laplacian_uniform(V, edges):
    """Uniform graph Laplacian L = D - A as a sparse tensor (V x V)."""
    e0, e1 = edges.unbind(1)
    idx = torch.cat([torch.stack([e0, e1]), torch.stack([e1, e0])], dim=1)
    vals = -torch.ones(idx.shape[1], device=edges.device)
    deg = torch.zeros(V, device=edges.device).index_add_(0, idx[0], torch.ones(idx.shape[1], device=edges.device))
    ii = torch.arange(V, device=edges.device)
    idx = torch.cat([idx, torch.stack([ii, ii])], dim=1)
    vals = torch.cat([vals, deg])
    return torch.sparse_coo_tensor(idx, vals, (V, V)).coalesce()


def face_normals_t(verts, faces):
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    n = torch.cross(v1 - v0, v2 - v0, dim=1)
    return n / n.norm(dim=1, keepdim=True).clamp_min(1e-12)


def vertex_normals_t(verts, faces):
    fn = face_normals_t(verts, faces)
    vn = torch.zeros_like(verts)
    for i in range(3):
        vn.index_add_(0, faces[:, i], fn)
    return vn / vn.norm(dim=1, keepdim=True).clamp_min(1e-12)


def gauss_kernel(device, channels=3):
    k = torch.tensor([[1., 4., 6., 4., 1.], [4., 16., 24., 16., 4.], [6., 24., 36., 24., 6.],
                      [4., 16., 24., 16., 4.], [1., 4., 6., 4., 1.]], device=device) / 256.0
    return k.repeat(channels, 1, 1, 1)


def _down_conv(x, kernel):
    """Reference implementation: 5x5 conv2d + decimation (cuDNN)."""
    return torch.nn.functional.conv2d(torch.nn.functional.pad(x, (2, 2, 2, 2), mode="reflect"), kernel, groups=x.shape[1])[:, :, ::2, ::2]


_TAPS = (1.0 / 16, 4.0 / 16, 6.0 / 16, 4.0 / 16, 1.0 / 16)


def _down(x, kernel=None):
    """Binomial 5-tap blur + 2x decimation of (B,C,H,W) as separable strided-slice arithmetic (memory-bound, a few ms for
    100 x 1536^2). Mathematically identical to _down_conv (gauss_kernel is the outer product of _TAPS); cuDNN chose a
    ~100x slower NHWC engine for that 1-channel convolution inside the training process (0.2 s per level)."""
    B, C, H, W = x.shape
    Ho, Wo = (H + 1) // 2, (W + 1) // 2
    xp = torch.nn.functional.pad(x, (2, 2, 2, 2), mode="reflect")
    v = None
    for k, w in enumerate(_TAPS):
        t = xp[:, :, k:k + 2 * Ho - 1:2, :] * w
        v = t if v is None else v + t
    out = None
    for k, w in enumerate(_TAPS):
        t = v[:, :, :, k:k + 2 * Wo - 1:2] * w
        out = t if out is None else out + t
    return out


def build_pyramid(target, kernel, levels=4):
    """The 4 down-sampled levels of a constant target (B,H,W,C), computed once (used by img_loss)."""
    tgt = target.permute(0, 3, 1, 2); out = []
    with torch.no_grad():
        for _ in range(levels):
            tgt = _down(tgt, kernel); out.append(tgt)
    return out


def img_loss(imgs, target, kernel, multi_scale=True, target_pyr=None):
    """imgs/target: (B,H,W,C). Multi-scale L2 as in NIE. target_pyr: optional precomputed build_pyramid(target)."""
    loss = (imgs - target).square().mean()
    if multi_scale:
        cur = imgs.permute(0, 3, 1, 2)
        tgt = target.permute(0, 3, 1, 2)
        for j in range(4):
            cur = _down(cur, kernel)
            tgt = target_pyr[j] if target_pyr is not None else _down(tgt, kernel)
            loss = loss + (cur - tgt).square().mean() / (j + 1)
    return loss


class Renderer:
    """Directional-light diffuse shading rendered with nvdiffrast (CUDA rasteriser), NIE-style cameras."""

    def __init__(self, num_views, res, seed=42, device="cuda", cam_dist=4.0, albedo=0.55):
        import nvdiffrast.torch as dr
        self.dr = dr
        self.res = res
        self.device = device
        self.glctx = dr.RasterizeCudaContext(device=device)
        self.albedo = albedo
        rng = np.random.RandomState(seed)
        proj = projection(x=0.5, n=1.5, f=100.0)
        mvs, lightdirs = [], []
        for _ in range(num_views):
            mv = translate(0, 0, -cam_dist) @ random_rotation(rng)
            mvs.append(mv)
            campos = np.linalg.inv(mv)[:3, 3]
            lightdirs.append(-campos / np.linalg.norm(campos))
        self.mvs = torch.as_tensor(np.stack(mvs), dtype=torch.float32, device=device)
        self.mvps = torch.as_tensor(proj, dtype=torch.float32, device=device) @ self.mvs
        self.lightdir = torch.as_tensor(np.stack(lightdirs), dtype=torch.float32, device=device)
        self.zero = torch.zeros((), device=device)
        self.kernel = gauss_kernel(device)

    def render(self, pos, faces, normals, view_ids=None):
        mvps = self.mvps if view_ids is None else self.mvps[view_ids]
        ld = self.lightdir if view_ids is None else self.lightdir[view_ids]
        v_hom = torch.nn.functional.pad(pos, (0, 1), "constant", 1.0)
        v_ndc = torch.matmul(v_hom, mvps.transpose(1, 2))
        rast, _ = self.dr.rasterize(self.glctx, v_ndc.contiguous(), faces.int().contiguous(), [self.res, self.res])
        pn, _ = self.dr.interpolate(normals[None].contiguous(), rast, faces.int().contiguous())
        diffuse = self.albedo * torch.sum(-ld.view(-1, 1, 1, 3) * pn, -1, keepdim=True)
        col = torch.where(rast[..., -1:] != 0, diffuse, self.zero)
        out = self.dr.antialias(col.contiguous(), rast, v_ndc.contiguous(), faces.int().contiguous())
        return torch.nan_to_num(out)

    def render_with_mask(self, pos, faces, normals, view_ids=None):
        """Shaded image and antialiased silhouette mask from ONE rasterisation (identical to render() + render_mask())."""
        mvps = self.mvps if view_ids is None else self.mvps[view_ids]
        ld = self.lightdir if view_ids is None else self.lightdir[view_ids]
        v_hom = torch.nn.functional.pad(pos, (0, 1), "constant", 1.0)
        v_ndc = torch.matmul(v_hom, mvps.transpose(1, 2)).contiguous()
        fi = faces.int().contiguous()
        rast, _ = self.dr.rasterize(self.glctx, v_ndc, fi, [self.res, self.res])
        pn, _ = self.dr.interpolate(normals[None].contiguous(), rast, fi)
        diffuse = self.albedo * torch.sum(-ld.view(-1, 1, 1, 3) * pn, -1, keepdim=True)
        hit = rast[..., -1:] != 0
        col = torch.where(hit, diffuse, self.zero)
        mask = torch.where(hit, torch.ones_like(rast[..., -1:]), self.zero)
        img = self.dr.antialias(col.contiguous(), rast, v_ndc, fi)
        msk = self.dr.antialias(mask.contiguous(), rast, v_ndc, fi)
        return torch.nan_to_num(img), msk

    def render_mask(self, pos, faces, view_ids=None):
        """Antialiased coverage mask (B,H,W,1) with silhouette gradients."""
        mvps = self.mvps if view_ids is None else self.mvps[view_ids]
        v_hom = torch.nn.functional.pad(pos, (0, 1), "constant", 1.0)
        v_ndc = torch.matmul(v_hom, mvps.transpose(1, 2))
        rast, _ = self.dr.rasterize(self.glctx, v_ndc.contiguous(), faces.int().contiguous(), [self.res, self.res])
        col = torch.where(rast[..., -1:] != 0, torch.ones_like(rast[..., -1:]), self.zero)
        return self.dr.antialias(col.contiguous(), rast, v_ndc.contiguous(), faces.int().contiguous())

    def render_mask_np(self, V, F, view_ids=None, batch=25):
        Vt = torch.as_tensor(V, dtype=torch.float32, device=self.device)
        Ft = torch.as_tensor(F, dtype=torch.long, device=self.device)
        n = self.mvps.shape[0] if view_ids is None else len(view_ids)
        ids = torch.arange(n, device=self.device) if view_ids is None else torch.as_tensor(view_ids, device=self.device)
        outs = []
        with torch.no_grad():
            for s in range(0, n, batch):
                outs.append(self.render_mask(Vt, Ft, ids[s:s + batch]))
        return torch.cat(outs)

    def render_mesh_np(self, V, F, view_ids=None, batch=25):
        Vt = torch.as_tensor(V, dtype=torch.float32, device=self.device)
        Ft = torch.as_tensor(F, dtype=torch.long, device=self.device)
        vn = vertex_normals_t(Vt, Ft)
        n = self.mvps.shape[0] if view_ids is None else len(view_ids)
        ids = torch.arange(n, device=self.device) if view_ids is None else torch.as_tensor(view_ids, device=self.device)
        outs = []
        with torch.no_grad():
            for s in range(0, n, batch):
                outs.append(self.render(Vt, Ft, vn, ids[s:s + batch]))
        return torch.cat(outs)


def psnr(imgs, target):
    mse = (imgs - target).square().mean()
    return float(-10.0 * torch.log10(mse.clamp_min(1e-12)))
