"""Batched GPU (PyTorch) reimplementation of OpenCV's DIS optical flow.

Every tensor carries a leading batch dimension B, so B independent flows
(different frame pairs, same image size) are computed per kernel launch, which
amortises the ~9000 tiny launches a single pair costs.  The pyramid, densify and
variational stages live here; the patch inverse search is the subclass's
(torch_dis_prop_fast.TorchDISBatchedPropFast), the only class instantiated.
"""
import math
import torch
import torch.nn.functional as F

EPS = 1e-3          # OpenCV EPS


# --------------------------------------------------------------------------
# Fusible variational leaves.  These are pure pointwise + fixed-stencil shift
# chains; Inductor (torch.compile) fuses each into a handful of kernels instead
# of the ~15 tiny elementwise/pad kernels the eager path launches per call.
# The math is byte-for-byte the eager path (fma contraction may differ in the
# last fp32 bit; validated to stay far below the flow tolerance).
# --------------------------------------------------------------------------
def _vr_setup_impl(Ix, Iy, Iz, Ixx, Ixy, Iyy, Ixz, Iyz, dWu, dWv, Wu2, Wv2,
                   Wu_r, Wu_l, Wu_d, Wu_u, Wv_r, Wv_l, Wv_d, Wv_u,
                   zeta2, eps2, gamma2, delta2, alpha2):
    dn = Ix * Ix + Iy * Iy + zeta2
    Ik1z = Iz + Ix * dWu + Iy * dWv
    wc = (delta2 / torch.sqrt(Ik1z * Ik1z / dn + eps2)) / dn
    a11 = wc * Ix * Ix + zeta2
    a12 = wc * Ix * Iy
    a22 = wc * Iy * Iy + zeta2
    b1 = -wc * Iz * Ix
    b2 = -wc * Iz * Iy
    dn1 = Ixx * Ixx + Ixy * Ixy + zeta2
    dn2 = Iyy * Iyy + Ixy * Ixy + zeta2
    Ik1zx = Ixz + Ixx * dWu + Ixy * dWv
    Ik1zy = Iyz + Ixy * dWu + Iyy * dWv
    wg = gamma2 / torch.sqrt(Ik1zx * Ik1zx / dn1 + Ik1zy * Ik1zy / dn2 + eps2)
    a11 = a11 + wg * (Ixx * Ixx / dn1 + Ixy * Ixy / dn2)
    a12 = a12 + wg * (Ixx * Ixy / dn1 + Ixy * Iyy / dn2)
    a22 = a22 + wg * (Ixy * Ixy / dn1 + Iyy * Iyy / dn2)
    b1 = b1 - wg * (Ixx * Ixz / dn1 + Ixy * Iyz / dn2)
    b2 = b2 - wg * (Ixy * Ixz / dn1 + Iyy * Iyz / dn2)
    cu = Wu2 + dWu; cv = Wv2 + dWv
    ux = F.pad(cu[..., 1:], (0, 1)) - cu; ux[..., -1] = 0          # shift_left(cu)-cu
    uy = F.pad(cu[..., 1:, :], (0, 0, 0, 1)) - cu; uy[..., -1, :] = 0  # shift_up(cu)-cu
    vx = F.pad(cv[..., 1:], (0, 1)) - cv; vx[..., -1] = 0
    vy = F.pad(cv[..., 1:, :], (0, 0, 0, 1)) - cv; vy[..., -1, :] = 0
    weight = alpha2 / torch.sqrt(ux * ux + vx * vx + uy * uy + vy * vy + eps2)
    wR = weight.clone(); wR[..., -1] = 0
    wB = weight.clone(); wB[..., -1, :] = 0
    wL = F.pad(wR[..., :-1], (1, 0))          # shift_right(wR)
    wT = F.pad(wB[..., :-1, :], (0, 0, 1, 0))  # shift_down(wB)
    s = wR + wL + wB + wT
    A11 = a11 + s
    A22 = a22 + s
    Lu = wR * (Wu_r - Wu2) + wL * (Wu_l - Wu2) + wB * (Wu_d - Wu2) + wT * (Wu_u - Wu2)
    Lv = wR * (Wv_r - Wv2) + wL * (Wv_l - Wv2) + wB * (Wv_d - Wv2) + wT * (Wv_u - Wv2)
    B1 = b1 + Lu
    B2 = b2 + Lv
    return a12, A11, A22, B1, B2, wL, wR, wT, wB


def _vr_sor_impl(dWu, dWv, wL, wR, wT, wB, a12, A11, A22, B1, B2, color, omega):
    sU = (wL * F.pad(dWu[..., :-1], (1, 0)) + wR * F.pad(dWu[..., 1:], (0, 1))
          + wT * F.pad(dWu[..., :-1, :], (0, 0, 1, 0))
          + wB * F.pad(dWu[..., 1:, :], (0, 0, 0, 1)))
    sV = (wL * F.pad(dWv[..., :-1], (1, 0)) + wR * F.pad(dWv[..., 1:], (0, 1))
          + wT * F.pad(dWv[..., :-1, :], (0, 0, 1, 0))
          + wB * F.pad(dWv[..., 1:, :], (0, 0, 0, 1)))
    du_new = dWu + omega * ((sU + B1 - dWv * a12) / A11 - dWu)
    dWu = torch.where(color, du_new, dWu)
    dv_new = dWv + omega * ((sV + B2 - dWu * a12) / A22 - dWv)
    dWv = torch.where(color, dv_new, dWv)
    return dWu, dWv


_VR_SETUP_C = None
_VR_SOR_C = None


def _compiled_leaves():
    global _VR_SETUP_C, _VR_SOR_C
    if _VR_SETUP_C is None:
        _VR_SETUP_C = torch.compile(_vr_setup_impl)
        _VR_SOR_C = torch.compile(_vr_sor_impl)
    return _VR_SETUP_C, _VR_SOR_C


def _stack_imgs(imgs, device):
    """list/array/tensor of B (H,W) uint8 images -> (B,1,H,W) float32."""
    if torch.is_tensor(imgs):
        t = imgs
    else:
        import numpy as np
        arr = np.asarray(imgs)
        t = torch.from_numpy(arr)
    t = t.to(device=device, dtype=torch.float32)
    if t.dim() == 2:            # single (H,W)
        t = t.view(1, 1, *t.shape)
    elif t.dim() == 3:          # (B,H,W)
        t = t.view(t.shape[0], 1, *t.shape[-2:])
    return t


class TorchDISBatched:
    def __init__(self, finest_scale=1, patch_size=8, patch_stride=3,
                 grad_descent_iter=25, vr_iter=10,
                 vr_alpha=20.0, vr_delta=5.0, vr_gamma=10.0,
                 vr_sor_iter=5, vr_omega=1.6, vr_zeta=0.1, vr_eps=1e-3,
                 border_size=16, device="cuda"):
        self.finest_scale = finest_scale
        self.ps = patch_size
        self.pstr = patch_stride
        self.gd_iter = grad_descent_iter
        self.vr_iter = vr_iter
        self.alpha = vr_alpha
        self.delta = vr_delta
        self.gamma = vr_gamma
        self.sor_iter = vr_sor_iter
        self.omega = vr_omega
        self.zeta = vr_zeta
        self.veps = vr_eps
        self.bsz = border_size
        self.device = device

        kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=device)
        self.sobelx = kx.view(1, 1, 3, 3)
        self.sobely = kx.t().contiguous().view(1, 1, 3, 3)

    # ---------------------------------------------------------------- pyramid
    def _coarsest_scale(self, H, W):
        a = int(math.log(max(H, W) / (4.0 * self.ps)) / math.log(2.0) + 0.5)
        b = int(math.log(min(H, W) / self.ps) / math.log(2.0))
        return min(a, b)

    @staticmethod
    def _area_half(img):
        out = F.avg_pool2d(img, 2, 2)
        return torch.floor(out + 0.5).clamp_(0, 255)

    def _spatial_gradient(self, img):
        p = F.pad(img, (1, 1, 1, 1), mode="reflect")
        gx = F.conv2d(p, self.sobelx)
        gy = F.conv2d(p, self.sobely)
        return torch.round(gx), torch.round(gy)

    def _build_pyramid(self, I0, I1):
        cs = self.coarsest_scale
        fs = self.finest_scale
        I0s, I1s = {}, {}
        cur0, cur1 = I0, I1
        for _ in range(fs):
            cur0 = self._area_half(cur0)
            cur1 = self._area_half(cur1)
        I0s[fs], I1s[fs] = cur0, cur1
        for i in range(fs + 1, cs + 1):
            cur0 = self._area_half(cur0)
            cur1 = self._area_half(cur1)
            I0s[i], I1s[i] = cur0, cur1
        return I0s, I1s

    # ------------------------------------------------------- structure tensor
    def _structure_tensor(self, I0x, I0y, hs, ws):
        ps, pstr = self.ps, self.pstr
        ones = torch.ones(1, 1, ps, ps, device=self.device)
        def boxsum(x):
            return F.conv2d(x, ones, stride=pstr)[:, 0]  # (B,hs,ws)
        Sxx = boxsum(I0x * I0x)
        Syy = boxsum(I0y * I0y)
        Sxy = boxsum(I0x * I0y)
        Sx = boxsum(I0x)
        Sy = boxsum(I0y)
        return (Sxx[:, :hs, :ws], Syy[:, :hs, :ws], Sxy[:, :hs, :ws],
                Sx[:, :hs, :ws], Sy[:, :hs, :ws])

    # --------------------------------------------------- patch inverse search
    def _densify(self, Sx, Sy, I0, I1):
        ps, pstr = self.ps, self.pstr
        B = I0.shape[0]
        h, w = I0.shape[-2:]
        hs, ws = Sx.shape[-2:]
        dev = self.device
        I0f = I0[:, 0]; I1f = I1[:, 0]              # (B,h,w)
        I1flat = I1f.reshape(B, h * w)
        ii = torch.arange(h, device=dev)
        jj = torch.arange(w, device=dev)
        end_is = torch.clamp(ii // pstr, max=hs - 1)
        start_is = torch.clamp(torch.clamp((ii - ps) // pstr + 1, min=0), max=end_is)
        end_js = torch.clamp(jj // pstr, max=ws - 1)
        start_js = torch.clamp(torch.clamp((jj - ps) // pstr + 1, min=0), max=end_js)
        I = ii.view(h, 1).float(); J = jj.view(1, w).float()
        sum_ux = torch.zeros(B, h, w, device=dev)
        sum_uy = torch.zeros(B, h, w, device=dev)
        sum_c = torch.zeros(B, h, w, device=dev)
        wm1 = float(w - 1) - EPS
        hm1 = float(h - 1) - EPS
        for di_s in range(3):
            is_row = end_is - di_s
            vi = is_row >= start_is
            is_c = is_row.clamp(0, hs - 1)
            for dj_s in range(3):
                js_col = end_js - dj_s
                vj = js_col >= start_js
                js_c = js_col.clamp(0, ws - 1)
                valid = (vi.view(1, h, 1) & vj.view(1, 1, w))   # (1,h,w)
                # gather sparse flow at (is_c row, js_c col) -> (B,h,w)
                u = Sx[:, is_c.view(h, 1).expand(h, w), js_c.view(1, w).expand(h, w)]
                v = Sy[:, is_c.view(h, 1).expand(h, w), js_c.view(1, w).expand(h, w)]
                jm = torch.clamp(J + u, 0.0, wm1)
                im = torch.clamp(I + v, 0.0, hm1)
                jl = torch.floor(jm); il = torch.floor(im)
                dj_ = jm - jl; di_ = im - il
                jl = jl.long(); il = il.long()
                ju = jl + 1; iu = il + 1
                # bilinear sample I1f (B,h,w) at (il/iu, jl/ju) via flat gather
                base_ll = (il * w + jl)
                base_lu = (il * w + ju)
                base_ul = (iu * w + jl)
                base_uu = (iu * w + ju)
                Ill = torch.gather(I1flat, 1, base_ll.reshape(B, -1)).view(B, h, w)
                Ilu = torch.gather(I1flat, 1, base_lu.reshape(B, -1)).view(B, h, w)
                Iul = torch.gather(I1flat, 1, base_ul.reshape(B, -1)).view(B, h, w)
                Iuu = torch.gather(I1flat, 1, base_uu.reshape(B, -1)).view(B, h, w)
                s = (dj_ * di_ * Iuu + (1 - dj_) * di_ * Iul
                     + dj_ * (1 - di_) * Ilu + (1 - dj_) * (1 - di_) * Ill)
                diff = s - I0f
                coef = 1.0 / torch.clamp(diff.abs(), min=1.0)
                coef = torch.where(valid, coef, torch.zeros_like(coef))
                sum_ux += coef * u
                sum_uy += coef * v
                sum_c += coef
        Ux = sum_ux / sum_c
        Uy = sum_uy / sum_c
        return Ux.view(B, 1, h, w), Uy.view(B, 1, h, w)

    # ------------------------------------------------- variational refinement
    @staticmethod
    def _sample_replicate(img, iy, ix):
        """Bilinear sample of img(B,1,h,w) at coords iy,ix (B,h,w), BORDER_REPLICATE."""
        B, _, h, w = img.shape
        I = img[:, 0]                       # (B,h,w)
        Iflat = I.reshape(B, h * w)
        x0 = torch.floor(ix); y0 = torch.floor(iy)
        dx = ix - x0; dy = iy - y0
        x0l = x0.long(); y0l = y0.long()
        x1l = x0l + 1; y1l = y0l + 1
        x0c = x0l.clamp(0, w - 1); x1c = x1l.clamp(0, w - 1)
        y0c = y0l.clamp(0, h - 1); y1c = y1l.clamp(0, h - 1)
        Ia = torch.gather(Iflat, 1, (y0c * w + x0c).reshape(B, -1)).view(B, h, w)
        Ib = torch.gather(Iflat, 1, (y0c * w + x1c).reshape(B, -1)).view(B, h, w)
        Ic = torch.gather(Iflat, 1, (y1c * w + x0c).reshape(B, -1)).view(B, h, w)
        Id = torch.gather(Iflat, 1, (y1c * w + x1c).reshape(B, -1)).view(B, h, w)
        return ((1 - dy) * (1 - dx) * Ia + (1 - dy) * dx * Ib
                + dy * (1 - dx) * Ic + dy * dx * Id)

    def _sobel1_x(self, img):
        p = F.pad(img, (1, 1, 0, 0), mode="replicate")
        return p[..., :, 2:] - p[..., :, :-2]

    def _sobel1_y(self, img):
        p = F.pad(img, (0, 0, 1, 1), mode="replicate")
        return p[..., 2:, :] - p[..., :-2, :]

    def _variational(self, I0, I1, Wu, Wv):
        if self.vr_iter <= 0:
            return Wu, Wv
        dev = self.device
        B = I0.shape[0]
        h, w = I0.shape[-2:]
        zeta2 = self.zeta ** 2
        eps2 = self.veps ** 2
        gamma2 = self.gamma / 2.0
        delta2 = self.delta / 2.0
        alpha2 = self.alpha / 2.0
        omega = self.omega

        I0f = I0[:, 0]                                          # (B,h,w)
        yy = torch.arange(h, device=dev).view(1, h, 1).float()
        xx = torch.arange(w, device=dev).view(1, 1, w).float()
        warped = self._sample_replicate(I1, yy + Wv[:, 0], xx + Wu[:, 0])
        avg = 0.5 * I0f + 0.5 * warped
        Iz = warped - I0f
        avg4 = avg.view(B, 1, h, w); Iz4 = Iz.view(B, 1, h, w)
        Ix = self._sobel1_x(avg4); Iy = self._sobel1_y(avg4)
        Ixz = self._sobel1_x(Iz4); Iyz = self._sobel1_y(Iz4)
        Ixx = self._sobel1_x(Ix); Ixy = self._sobel1_y(Ix); Iyy = self._sobel1_y(Iy)
        Ix = Ix[:, 0]; Iy = Iy[:, 0]; Iz = Iz4[:, 0]
        Ixz = Ixz[:, 0]; Iyz = Iyz[:, 0]
        Ixx = Ixx[:, 0]; Ixy = Ixy[:, 0]; Iyy = Iyy[:, 0]

        Wu2 = Wu[:, 0]; Wv2 = Wv[:, 0]                          # (B,h,w)
        dWu = torch.zeros(B, h, w, device=dev)
        dWv = torch.zeros(B, h, w, device=dev)

        ii = torch.arange(h, device=dev).view(h, 1)
        jj = torch.arange(w, device=dev).view(1, w)
        red = ((ii + jj) % 2 == 0)                             # (h,w) broadcasts over B
        black = ~red

        def shift_left(a):   # a[...,i,j+1]
            return F.pad(a[..., 1:], (0, 1))
        def shift_right(a):  # a[...,i,j-1]
            return F.pad(a[..., :-1], (1, 0))
        def shift_up(a):     # a[...,i+1,j]
            return F.pad(a[..., 1:, :], (0, 0, 0, 1))
        def shift_down(a):   # a[...,i-1,j]
            return F.pad(a[..., :-1, :], (0, 0, 1, 0))

        Wu_r = shift_left(Wu2); Wu_l = shift_right(Wu2)
        Wu_d = shift_up(Wu2); Wu_u = shift_down(Wu2)
        Wv_r = shift_left(Wv2); Wv_l = shift_right(Wv2)
        Wv_d = shift_up(Wv2); Wv_u = shift_down(Wv2)

        setup_c, sor_c = _compiled_leaves()
        for _ in range(self.vr_iter):
            a12, A11, A22, B1, B2, wL, wR, wT, wB = setup_c(
                Ix, Iy, Iz, Ixx, Ixy, Iyy, Ixz, Iyz, dWu, dWv, Wu2, Wv2,
                Wu_r, Wu_l, Wu_d, Wu_u, Wv_r, Wv_l, Wv_d, Wv_u,
                zeta2, eps2, gamma2, delta2, alpha2)
            for _s in range(self.sor_iter):
                for color in (red, black):
                    dWu, dWv = sor_c(dWu, dWv, wL, wR, wT, wB, a12, A11, A22,
                                     B1, B2, color, omega)

        return (Wu2 + dWu).view(B, 1, h, w), (Wv2 + dWv).view(B, 1, h, w)

    # --------------------------------------------------------------- calc
    @torch.no_grad()
    def calc(self, I0, I1):
        dev = self.device
        I0 = _stack_imgs(I0, dev); I1 = _stack_imgs(I1, dev)
        B = I0.shape[0]
        H, W = I0.shape[-2:]
        self.coarsest_scale = self._coarsest_scale(H, W)
        cs, fs = self.coarsest_scale, self.finest_scale

        I0s, I1s = self._build_pyramid(I0, I1)

        Ux = Uy = None
        for i in range(cs, fs - 1, -1):
            I0i = I0s[i]; I1i = I1s[i]
            h, w = I0i.shape[-2:]
            I1ext = F.pad(I1i, (self.bsz,) * 4, mode="replicate")
            I0x, I0y = self._spatial_gradient(I0i)
            if Ux is None:
                Ux = torch.zeros(B, 1, h, w, device=dev)
                Uy = torch.zeros(B, 1, h, w, device=dev)
            Sx, Sy = self._inverse_search(I0i, I1ext, I0x, I0y, Ux, Uy, i)
            Ux, Uy = self._densify(Sx, Sy, I0i, I1i)
            Ux, Uy = self._variational(I0i, I1i, Ux, Uy)
            if i > fs:
                nh, nw = I0s[i - 1].shape[-2:]
                Ux = F.interpolate(Ux, size=(nh, nw), mode="bilinear", align_corners=False) * 2.0
                Uy = F.interpolate(Uy, size=(nh, nw), mode="bilinear", align_corners=False) * 2.0

        Ux = F.interpolate(Ux, size=(H, W), mode="bilinear", align_corners=False) * (1 << fs)
        Uy = F.interpolate(Uy, size=(H, W), mode="bilinear", align_corners=False) * (1 << fs)
        # (B,H,W,2)
        flow = torch.stack([Ux[:, 0], Uy[:, 0]], dim=-1)
        return flow
