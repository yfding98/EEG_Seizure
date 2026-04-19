#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MultiScaleBrainNetworkExtractor -- 多尺度脑网络特征提取

从 EEG 补丁计算 4 种导联间连通性矩阵:
  1. GC   (Granger Causality)             非对称
  2. TE   (Transfer Entropy, binned)      非对称
  3. AEC  (Asymmetric Envelope Corr.)     非对称
  4. wPLI (weighted Phase Lag Index)       对称

所有特征归一化到 [0, 1], NaN / 零通道自动处理.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

# TCP 22 channel labels
TCP_NAMES = [
    'FP1-F7', 'F7-T3', 'T3-T5', 'T5-O1',
    'FP2-F8', 'F8-T4', 'T4-T6', 'T6-O2',
    'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
    'A1-T3',  'T3-C3', 'C3-CZ', 'CZ-C4',
    'C4-T4',  'T4-A2',
]

DEFAULT_BANDS = {
    'delta': (1.0, 4.0),
    'theta': (4.0, 8.0),
    'alpha': (8.0, 13.0),
    'beta':  (13.0, 30.0),
    'gamma': (30.0, 80.0),
}


# =====================================================================
# Helpers
# =====================================================================

def _hilbert_torch(x: torch.Tensor) -> torch.Tensor:
    """FFT Hilbert -> analytic signal.  x: [..., T] real -> [..., T] complex."""
    N = x.shape[-1]
    Xf = torch.fft.fft(x, dim=-1)
    h = torch.zeros(N, device=x.device, dtype=x.dtype)
    h[0] = 1.0
    if N % 2 == 0:
        h[1:N // 2] = 2.0
        h[N // 2] = 1.0
    else:
        h[1:(N + 1) // 2] = 2.0
    return torch.fft.ifft(Xf * h, dim=-1)


def _bandpass_fft(x: torch.Tensor, fs: float, lo: float, hi: float) -> torch.Tensor:
    """Brick-wall FFT bandpass.  x: [..., T] -> [..., T]."""
    N = x.shape[-1]
    freqs = torch.fft.rfftfreq(N, d=1.0 / fs, device=x.device)
    mask = ((freqs >= lo) & (freqs <= hi)).float()
    return torch.fft.irfft(torch.fft.rfft(x, dim=-1) * mask, n=N, dim=-1)


def _norm01(m: torch.Tensor) -> torch.Tensor:
    """Per-matrix min-max to [0,1].  m: [..., C, C]."""
    s = m.shape
    f = m.reshape(-1, s[-2] * s[-1])
    lo = f.min(-1, keepdim=True).values
    hi = f.max(-1, keepdim=True).values
    return ((f - lo) / (hi - lo).clamp(min=1e-10)).reshape(s)


def _safe(m: torch.Tensor) -> torch.Tensor:
    """NaN/Inf -> 0."""
    return torch.where(torch.isfinite(m), m, torch.zeros_like(m))


def _ch_mask(x: torch.Tensor) -> torch.Tensor:
    """[N,C,T] -> [N,C] bool (True=valid)."""
    return x.abs().sum(-1) > 0


def _apply_ch_mask(mat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Zero rows/cols of invalid channels. mat:[N,C,C], mask:[N,C]."""
    r = mask.unsqueeze(-1).float()
    c = mask.unsqueeze(-2).float()
    return mat * r * c


def _batched_bincount(idx: torch.Tensor, nb: int) -> torch.Tensor:
    """idx:[N,T] long -> counts:[N,nb]."""
    N, T = idx.shape
    out = torch.zeros(N, nb, device=idx.device)
    out.scatter_add_(1, idx, torch.ones(N, T, device=idx.device))
    return out


def _entropy(counts: torch.Tensor) -> torch.Tensor:
    """Shannon entropy from counts. counts:[N,nb] -> H:[N]."""
    total = counts.sum(-1, keepdim=True).clamp(min=1)
    p = counts / total
    lp = torch.where(p > 0, p.log(), torch.zeros_like(p))
    return -(p * lp).sum(-1)


# =====================================================================
# Module
# =====================================================================

class MultiScaleBrainNetworkExtractor(nn.Module):
    """
    多尺度脑网络特征提取器

    Parameters
    ----------
    n_channels : int       通道数 (22)
    patch_len  : int       补丁采样点 (100)
    fs         : float     采样率 (200)
    gc_order   : int       GC VAR 阶数 (20)
    te_n_bins  : int       TE 直方图 bins (8)
    te_lag     : int       TE 延迟 (1)
    gc_ridge   : float     GC 岭正则化 (1e-3)
    """

    def __init__(
        self,
        n_channels: int = 22,
        patch_len: int = 100,
        fs: float = 200.0,
        gc_order: int = 20,
        te_n_bins: int = 8,
        te_lag: int = 1,
        gc_ridge: float = 1e-3,
        bands: Optional[Dict[str, Tuple[float, float]]] = None,
        channel_names: Optional[List[str]] = None,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.patch_len = patch_len
        self.fs = fs
        self.gc_order = gc_order
        self.te_n_bins = te_n_bins
        self.te_lag = te_lag
        self.gc_ridge = gc_ridge
        self.bands = bands or DEFAULT_BANDS
        self.channel_names = channel_names or TCP_NAMES[:n_channels]
        self._last_result: Optional[Dict] = None

    # -----------------------------------------------------------------
    # wPLI  (fully vectorized, symmetric)
    # -----------------------------------------------------------------
    @torch.no_grad()
    def compute_wpli(self, x: torch.Tensor) -> torch.Tensor:
        """x:[N,C,L] -> wpli:[N,C,C] in [0,1]."""
        analytic = _hilbert_torch(x)                         # [N,C,L] complex
        ai = analytic.unsqueeze(2)                           # [N,C,1,L]
        aj = analytic.unsqueeze(1).conj()                    # [N,1,C,L]
        im_csd = (ai * aj).imag                              # [N,C,C,L]
        num = im_csd.mean(-1).abs()                          # |E[Im]|
        den = im_csd.abs().mean(-1).clamp(min=1e-10)         # E[|Im|]
        return _safe(num / den)

    # -----------------------------------------------------------------
    # AEC  (orthogonalized, asymmetric)
    # -----------------------------------------------------------------
    @torch.no_grad()
    def compute_aec(self, x: torch.Tensor) -> torch.Tensor:
        """x:[N,C,L] -> aec:[N,C,C], aec[i,j]=AEC(i->j)."""
        analytic = _hilbert_torch(x)                         # [N,C,L] complex
        ai = analytic.unsqueeze(2)                           # [N,C,1,L] src
        aj = analytic.unsqueeze(1)                           # [N,1,C,L] tgt
        # project j onto i, then orthogonalize
        dot_ji = (aj * ai.conj()).sum(-1)                    # [N,C,C]
        dot_ii = (ai * ai.conj()).sum(-1).real.clamp(min=1e-10)
        proj = dot_ji / dot_ii                               # [N,C,C]
        aj_orth = aj - proj.unsqueeze(-1) * ai               # [N,C,C,L]
        env_orth = aj_orth.abs()                             # [N,C,C,L]
        ei = analytic.abs().unsqueeze(2)                     # [N,C,1,L]
        # pearson corr
        ei_d = ei - ei.mean(-1, keepdim=True)
        eo_d = env_orth - env_orth.mean(-1, keepdim=True)
        num = (ei_d * eo_d).sum(-1)
        den = (ei_d.norm(dim=-1) * eo_d.norm(dim=-1)).clamp(min=1e-10)
        return _safe((num / den).abs())

    # -----------------------------------------------------------------
    # Granger Causality  (batched VAR, loop over target channel)
    # -----------------------------------------------------------------
    @torch.no_grad()
    def compute_gc(self, x: torch.Tensor) -> torch.Tensor:
        """x:[N,C,L] -> gc:[N,C,C], gc[i,j]=GC(j->i) >= 0."""
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        N, C, L = x.shape
        p = self.gc_order
        T = L - p
        dev = x.device
        if T <= 2 * p:
            return torch.zeros(N, C, C, device=dev, dtype=orig_dtype)

        # lagged matrix: [N,C,T,p]
        lags = torch.stack(
            [x[:, :, p - k - 1: L - k - 1] for k in range(p)], dim=-1
        )
        targets = x[:, :, p:]                                # [N,C,T]

        # -- restricted residual variance (AR on own lags) --
        Xr = lags.reshape(N * C, T, p)
        yr = targets.reshape(N * C, T)
        XtX_r = Xr.transpose(-2, -1) @ Xr
        XtX_r += self.gc_ridge * torch.eye(p, device=dev)
        Xty_r = Xr.transpose(-2, -1) @ yr.unsqueeze(-1)
        beta_r = torch.linalg.lstsq(XtX_r, Xty_r).solution
        res_r = yr - (Xr @ beta_r).squeeze(-1)
        var_r = res_r.var(-1).reshape(N, C)                  # [N,C]

        # -- unrestricted (target i, all source j) --
        gc = torch.zeros(N, C, C, device=dev)
        reg_u = self.gc_ridge * torch.eye(2 * p, device=dev)

        for i in range(C):
            li = lags[:, i: i + 1].expand(-1, C, -1, -1)    # [N,C,T,p]
            Xu = torch.cat([li, lags], dim=-1)               # [N,C,T,2p]
            Xu = Xu.reshape(N * C, T, 2 * p)
            yi = targets[:, i].unsqueeze(1).expand(-1, C, -1).reshape(N * C, T)
            XtX_u = Xu.transpose(-2, -1) @ Xu + reg_u
            Xty_u = Xu.transpose(-2, -1) @ yi.unsqueeze(-1)
            beta_u = torch.linalg.lstsq(XtX_u, Xty_u).solution
            res_u = yi - (Xu @ beta_u).squeeze(-1)
            var_u = res_u.var(-1).reshape(N, C).clamp(min=1e-10)
            gc[:, i, :] = torch.log(var_r[:, i: i + 1] / var_u).clamp(min=0)
            gc[:, i, i] = 0.0

        return _safe(gc).to(orig_dtype)

    # -----------------------------------------------------------------
    # Transfer Entropy  (binned, batched)
    # -----------------------------------------------------------------
    @torch.no_grad()
    def compute_te(self, x: torch.Tensor) -> torch.Tensor:
        """x:[N,C,L] -> te:[N,C,C], te[i,j]=TE(j->i) >= 0."""
        N, C, L = x.shape
        nb = self.te_n_bins
        lag = self.te_lag
        dev = x.device
        T = L - lag
        if T < 4:
            return torch.zeros(N, C, C, device=dev)

        # discretize
        xmin = x.min(-1, keepdim=True).values
        xmax = x.max(-1, keepdim=True).values
        xn = (x - xmin) / (xmax - xmin).clamp(min=1e-10)
        xd = (xn * (nb - 1)).long().clamp(0, nb - 1)        # [N,C,L]

        y_fut = xd[:, :, lag:]                               # [N,C,T]
        y_pst = xd[:, :, :T]                                # [N,C,T]

        te = torch.zeros(N, C, C, device=dev)

        for i in range(C):
            yf = y_fut[:, i]                                 # [N,T]
            yp = y_pst[:, i]                                 # [N,T]

            # H(yf, yp) -- same for all j
            idx_ab = yf * nb + yp                            # [N,T]
            H_ab = _entropy(_batched_bincount(idx_ab, nb * nb))

            # H(yp) -- same for all j
            H_b = _entropy(_batched_bincount(yp, nb))

            # expand for all j
            yf_e = yf.unsqueeze(1).expand(-1, C, -1)        # [N,C,T]
            yp_e = yp.unsqueeze(1).expand(-1, C, -1)        # [N,C,T]
            xp_all = y_pst                                  # [N,C,T]

            # H(yp, xp)  [N*C]
            idx_bc = (yp_e * nb + xp_all).reshape(N * C, T)
            H_bc = _entropy(_batched_bincount(idx_bc, nb * nb)).reshape(N, C)

            # H(yf, yp, xp)  [N*C]
            idx_abc = (yf_e * nb * nb + yp_e * nb + xp_all).reshape(N * C, T)
            H_abc = _entropy(_batched_bincount(idx_abc, nb ** 3)).reshape(N, C)

            te[:, i, :] = (H_ab.unsqueeze(-1) + H_bc - H_b.unsqueeze(-1) - H_abc).clamp(min=0)
            te[:, i, i] = 0.0

        return _safe(te)

    # -----------------------------------------------------------------
    # Internal: compute all 4 features on pre-shaped [N,C,L] data
    # -----------------------------------------------------------------
    def _compute_all(
        self, x: torch.Tensor, B: int, P: int,
    ) -> Dict[str, torch.Tensor]:
        mask = _ch_mask(x)                                   # [N,C]

        gc  = _apply_ch_mask(self.compute_gc(x),  mask)
        te  = _apply_ch_mask(self.compute_te(x),  mask)
        aec = _apply_ch_mask(self.compute_aec(x), mask)
        wpli= _apply_ch_mask(self.compute_wpli(x),mask)

        # normalize
        gc   = _norm01(gc)
        te   = _norm01(te)
        aec  = _norm01(aec)
        wpli = _norm01(wpli)

        C = x.shape[1]
        gc   = gc.reshape(B, P, C, C)
        te   = te.reshape(B, P, C, C)
        aec  = aec.reshape(B, P, C, C)
        wpli = wpli.reshape(B, P, C, C)
        stacked = torch.stack([gc, te, aec, wpli], dim=-1)   # [B,P,C,C,4]

        return {'gc': gc, 'te': te, 'aec': aec, 'wpli': wpli, 'all': stacked}

    # -----------------------------------------------------------------
    # forward
    # -----------------------------------------------------------------
    def forward(
        self,
        patches: torch.Tensor,
        by_band: bool = False,
    ) -> Dict:
        """
        Args
        ----
        patches : [B, P, C, L]
        by_band : if True, return {band_name: {feature: tensor}}

        Returns
        -------
        dict with keys 'gc','te','aec','wpli','all'
        (or nested dict when by_band=True)
        """
        B, P, C, L = patches.shape
        x = patches.reshape(B * P, C, L)

        if not by_band:
            result = self._compute_all(x, B, P)
            self._last_result = result
            return result

        band_results: Dict[str, Dict] = {}
        for name, (lo, hi) in self.bands.items():
            xb = _bandpass_fft(x, self.fs, lo, hi)
            band_results[name] = self._compute_all(xb, B, P)
        self._last_result = band_results
        return band_results

    # -----------------------------------------------------------------
    # Visualization
    # -----------------------------------------------------------------
    def visualize_network(
        self,
        batch_idx: int,
        patch_idx: int,
        feature: str = 'wpli',
        result: Optional[Dict] = None,
        save_path: Optional[str] = None,
    ):
        """Plot a single [C,C] connectivity matrix as a heatmap."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for visualization")

        r = result or self._last_result
        if r is None:
            raise RuntimeError("No result cached -- call forward() first or pass result=")

        mat = r[feature][batch_idx, patch_idx].detach().cpu().numpy()
        C = mat.shape[0]
        names = self.channel_names[:C]

        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(mat, cmap='hot', vmin=0, vmax=1, aspect='equal')
        ax.set_xticks(range(C))
        ax.set_yticks(range(C))
        ax.set_xticklabels(names, rotation=90, fontsize=7)
        ax.set_yticklabels(names, fontsize=7)
        ax.set_title(f'{feature.upper()}  [sample {batch_idx}, patch {patch_idx}]')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        return fig

    def extra_repr(self) -> str:
        return (
            f"C={self.n_channels}, L={self.patch_len}, fs={self.fs}, "
            f"gc_order={self.gc_order}, te_bins={self.te_n_bins}"
        )


# =====================================================================
# Self-test
# =====================================================================

def _test():
    torch.manual_seed(0)
    B, P, C, L = 2, 3, 22, 100
    patches = torch.randn(B, P, C, L)

    mod = MultiScaleBrainNetworkExtractor(n_channels=C, patch_len=L, fs=200.0)
    print(mod)

    # --- basic forward ---
    result = mod(patches)
    for k in ('gc', 'te', 'aec', 'wpli'):
        t = result[k]
        print(f"  {k:5s}: shape={list(t.shape)}, "
              f"min={t.min():.4f}, max={t.max():.4f}")
        assert t.shape == (B, P, C, C), f"{k} shape mismatch"
        assert torch.isfinite(t).all(), f"{k} has non-finite values"
        assert t.min() >= 0 and t.max() <= 1.001, f"{k} out of [0,1]"

    assert result['all'].shape == (B, P, C, C, 4)

    # --- symmetry check for wPLI ---
    w = result['wpli'][0, 0]
    assert torch.allclose(w, w.T, atol=1e-5), "wPLI should be symmetric"

    # --- asymmetry check for GC ---
    g = result['gc'][0, 0]
    diff = (g - g.T).abs().sum()
    # GC should not be perfectly symmetric (in general)
    print(f"  GC asymmetry: {diff:.4f}")

    # --- zero-channel handling ---
    patches_z = patches.clone()
    patches_z[:, :, 5, :] = 0.0  # zero out channel 5
    rz = mod(patches_z)
    for k in ('gc', 'te', 'aec', 'wpli'):
        assert rz[k][:, :, 5, :].abs().sum() == 0, f"{k} row 5 should be 0"
        assert rz[k][:, :, :, 5].abs().sum() == 0, f"{k} col 5 should be 0"

    # --- by_band ---
    rb = mod(patches, by_band=True)
    assert set(rb.keys()) == {'delta', 'theta', 'alpha', 'beta', 'gamma'}
    for band_name, bd in rb.items():
        assert bd['all'].shape == (B, P, C, C, 4), f"{band_name} stacked shape"

    print("\n[PASS] All tests passed!")


if __name__ == '__main__':
    _test()
