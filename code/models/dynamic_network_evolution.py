#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DynamicNetworkEvolutionModel -- 发作对齐的动态网络演化建模

Pipeline:
  brain_networks [B, P, 22, 22, 4]  (gc/te/aec/wpli)
       |
  (a) Multi-Branch Snapshot Encoder
      ├─ GC  branch (DirectedGAT) ─┐
      ├─ TE  branch (DirectedGAT) ─┤ CrossBranchAttention → [B,P,128]
      ├─ AEC branch (DirectedGAT) ─┤    + branch_weights [B,P,4]
      └─ wPLI branch (GCN)        ─┘
       |
  (b) BiGRU Evolution       -> [B, P, 256]
       |
  +----+----+
  |         |
  (c) Transition Detector   (d) Pattern Classifier
  [B, P] probs              [B, 3] logits
"""

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# Feature index mapping (order in last dim of brain_networks)
FEATURE_NAMES = ['gc', 'te', 'aec', 'wpli']
FEATURE_SYMMETRIC = [False, False, False, True]  # only wPLI is symmetric


# =====================================================================
# Graph Layers
# =====================================================================

class GCNLayer(nn.Module):
    """Undirected GCN: h' = sigma(D^{-1/2} A D^{-1/2} h W).  For symmetric adj (wPLI)."""

    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """h:[..., N, D_in], adj:[..., N, N] -> [..., N, D_out]."""
        deg = adj.sum(-1).clamp(min=1e-6)
        deg_inv_sqrt = deg.pow(-0.5)
        norm_adj = adj * deg_inv_sqrt.unsqueeze(-1) * deg_inv_sqrt.unsqueeze(-2)
        return self.W(torch.matmul(norm_adj, h))


class DirectedGATLayer(nn.Module):
    """
    Directed Graph Attention for asymmetric adj (GC, TE, AEC).

    attention(i,j) = LeakyReLU(a^T [Wh_i || Wh_j]) * adj(j,i)
    Only neighbours with non-zero adj contribute.
    """

    def __init__(self, in_dim: int, out_dim: int, n_heads: int = 4,
                 dropout: float = 0.1, negative_slope: float = 0.2):
        super().__init__()
        assert out_dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = out_dim // n_heads
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Parameter(torch.empty(n_heads, 2 * self.head_dim))
        nn.init.xavier_uniform_(self.a.unsqueeze(0))
        self.leaky = nn.LeakyReLU(negative_slope)
        self.drop = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """h:[..., N, D_in], adj:[..., N, N] -> [..., N, D_out]."""
        lead = h.shape[:-2]
        N = h.shape[-2]
        Wh = self.W(h)  # [..., N, out]
        Wh = Wh.reshape(*lead, N, self.n_heads, self.head_dim)  # [...,N,H,Hd]

        # pairwise attention scores
        Wh_i = Wh.unsqueeze(-3)  # [..., 1, N, H, Hd]
        Wh_j = Wh.unsqueeze(-4)  # [..., N, 1, H, Hd]
        cat_ij = torch.cat([Wh_i.expand(*lead, N, N, self.n_heads, self.head_dim),
                            Wh_j.expand(*lead, N, N, self.n_heads, self.head_dim)], dim=-1)
        e = (cat_ij * self.a).sum(-1)  # [..., N, N, H]
        e = self.leaky(e)

        # mask by adjacency (directed: adj[j,i] > 0 means j->i edge)
        mask = (adj.unsqueeze(-1) > 1e-8)  # [..., N, N, 1]
        e = e.masked_fill(~mask, -1e9)
        alpha = torch.softmax(e, dim=-3)  # softmax over source dim
        alpha = self.drop(alpha)

        # aggregate: h_i' = sum_j alpha(j,i) * Wh_j
        # alpha: [..., N_src, N_tgt, H], Wh: [..., N, H, Hd]
        Wh_src = Wh.unsqueeze(-3)  # [..., N, 1, H, Hd]
        out = (alpha.unsqueeze(-1) * Wh_src).sum(dim=-4)  # [..., N, H, Hd]
        return out.reshape(*lead, N, self.n_heads * self.head_dim)


# =====================================================================
# Single Branch Encoder
# =====================================================================

class SingleBranchEncoder(nn.Module):
    """One branch: 2-layer graph network + global mean pool -> [branch_dim]."""

    def __init__(self, n_channels: int, hidden: int = 32,
                 symmetric: bool = False, dropout: float = 0.1):
        super().__init__()
        self.symmetric = symmetric
        if symmetric:
            self.layer1 = GCNLayer(n_channels, hidden)
            self.layer2 = GCNLayer(hidden, hidden)
        else:
            self.layer1 = DirectedGATLayer(n_channels, hidden, n_heads=4, dropout=dropout)
            self.layer2 = DirectedGATLayer(hidden, hidden, n_heads=4, dropout=dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, node_feat: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """node_feat:[*, N, D], adj:[*, N, N] -> [*, hidden]."""
        h = F.relu(self.layer1(node_feat, adj))
        h = self.drop(h)
        h = F.relu(self.layer2(h, adj))   # [*, N, hidden]
        return h.mean(dim=-2)              # [*, hidden]  global pool


# =====================================================================
# Multi-Branch Snapshot Encoder
# =====================================================================

class MultiBranchSnapshotEncoder(nn.Module):
    """
    Encode [22, 22, 4] brain-network via 4 parallel graph branches
    + cross-branch MultiheadAttention fusion.

    Branches:
      - GC   (idx 0): DirectedGAT  (asymmetric)
      - TE   (idx 1): DirectedGAT  (asymmetric)
      - AEC  (idx 2): DirectedGAT  (asymmetric)
      - wPLI (idx 3): GCN          (symmetric)

    Each branch: 2-layer graph (32 hidden) + MeanPool -> 32
    Fusion: 4 branch tokens -> MultiheadAttention -> weighted sum -> 128

    Returns:
      features   : [*, out_dim]
      branch_wts : [*, 4]
    """

    def __init__(
        self,
        n_channels: int = 22,
        n_features: int = 4,
        branch_hidden: int = 32,
        out_dim: int = 128,
        n_attn_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_features = n_features
        self.out_dim = out_dim

        # one branch per feature
        self.branches = nn.ModuleList([
            SingleBranchEncoder(
                n_channels=n_channels,
                hidden=branch_hidden,
                symmetric=FEATURE_SYMMETRIC[i],
                dropout=dropout,
            )
            for i in range(n_features)
        ])

        # cross-branch attention
        self.branch_proj = nn.Linear(branch_hidden, out_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=out_dim, num_heads=n_attn_heads,
            dropout=dropout, batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(out_dim)
        self.attn_gate = nn.Sequential(
            nn.Linear(out_dim, out_dim // 2),
            nn.GELU(),
            nn.Linear(out_dim // 2, 1),
        )
        self.norm = nn.LayerNorm(out_dim)
        self.register_buffer(
            'active_feature_mask',
            torch.ones(n_features, dtype=torch.float32),
            persistent=False,
        )
        self.set_active_features(FEATURE_NAMES[:n_features])

    def set_active_features(self, active_features: Sequence[str]) -> None:
        normalized = tuple(str(name).lower() for name in active_features)
        mask = torch.tensor(
            [1.0 if name in normalized else 0.0 for name in FEATURE_NAMES[:self.n_features]],
            dtype=self.active_feature_mask.dtype,
            device=self.active_feature_mask.device,
        )
        if mask.sum() <= 0:
            raise ValueError("MultiBranchSnapshotEncoder requires at least one active feature")
        self.active_feature_mask.copy_(mask)

    def forward(
        self, nets: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        nets: [*, 22, 22, 4]
        Returns:
          features   : [*, out_dim]
          branch_wts : [*, 4]
        """
        lead = nets.shape[:-3]
        C = nets.shape[-3]
        flat = nets.reshape(-1, C, C, self.n_features)  # [N, C, C, F]
        N = flat.shape[0]
        active_mask = self.active_feature_mask.to(device=flat.device, dtype=flat.dtype)

        branch_outs = []  # will be [N, F, branch_hidden]
        for f_idx, branch in enumerate(self.branches):
            adj_f = flat[..., f_idx]                # [N, C, C]
            node_f = flat[..., f_idx]               # [N, C, C] -> use row as node feat
            h_f = branch(node_f, adj_f)             # [N, branch_hidden]
            h_f = h_f * active_mask[f_idx]
            branch_outs.append(h_f)

        tokens = torch.stack(branch_outs, dim=1)    # [N, 4, branch_hidden]
        tokens = self.branch_proj(tokens)            # [N, 4, out_dim]
        tokens = tokens * active_mask.view(1, self.n_features, 1)

        # cross-branch self-attention
        key_padding_mask = active_mask.unsqueeze(0).expand(N, -1) <= 0
        has_inactive = bool(key_padding_mask.any().item())
        attn_out, _ = self.cross_attn(
            tokens, tokens, tokens,
            key_padding_mask=key_padding_mask if has_inactive else None,
        )  # [N, 4, out_dim]
        attn_out = attn_out + tokens                 # residual
        attn_out = attn_out * active_mask.view(1, self.n_features, 1)

        # branch importance weights
        gate_logits = self.attn_gate(self.attn_norm(attn_out)).squeeze(-1)  # [N, 4]
        if has_inactive:
            gate_logits = gate_logits.masked_fill(key_padding_mask, -1e9)
        branch_wts = torch.softmax(gate_logits, dim=-1)     # [N, 4]
        branch_wts = branch_wts * active_mask.view(1, -1)
        branch_wts = branch_wts / branch_wts.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        # weighted fusion
        fused = (attn_out * branch_wts.unsqueeze(-1)).sum(dim=1)  # [N, out_dim]
        fused = self.norm(fused)

        return fused.reshape(*lead, self.out_dim), branch_wts.reshape(*lead, self.n_features)


# =====================================================================
# Main Model
# =====================================================================

class DynamicNetworkEvolutionModel(nn.Module):
    """
    发作对齐的动态网络演化建模

    Parameters
    ----------
    n_channels       : int   导联数 (22)
    n_net_features   : int   网络特征数 (4: gc,te,aec,wpli)
    max_patches      : int   最大补丁数 (20)
    gcn_hidden       : int   GCN 隐藏维度 (64)
    snapshot_dim     : int   快照编码维度 (128)
    gru_hidden       : int   GRU 隐藏维度 (128)
    gru_layers       : int   GRU 层数 (2)
    gru_dropout      : float GRU dropout (0.2)
    transition_window: tuple 转变检测窗口 (-2s, +3s)
    n_patterns       : int   重组模式数 (3)
    use_checkpoint   : bool  梯度检查点 (False)
    """

    def __init__(
        self,
        n_channels: int = 22,
        n_net_features: int = 4,
        max_patches: int = 20,
        gcn_hidden: int = 64,
        snapshot_dim: int = 128,
        gru_hidden: int = 128,
        gru_layers: int = 2,
        gru_dropout: float = 0.2,
        transition_window: Tuple[float, float] = (-2.0, 3.0),
        n_patterns: int = 3,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.max_patches = max_patches
        self.snapshot_dim = snapshot_dim
        self.gru_hidden = gru_hidden
        self.transition_window = transition_window
        self.n_patterns = n_patterns
        self.use_checkpoint = use_checkpoint

        # (a) Multi-branch snapshot encoder
        self.snapshot_enc = MultiBranchSnapshotEncoder(
            n_channels=n_channels,
            n_features=n_net_features,
            branch_hidden=gcn_hidden // 2,   # 32 per branch
            out_dim=snapshot_dim,
            dropout=0.1,
        )

        # time embedding: project scalar relative time -> snapshot_dim
        self.time_embed = nn.Sequential(
            nn.Linear(1, snapshot_dim // 2),
            nn.GELU(),
            nn.Linear(snapshot_dim // 2, snapshot_dim),
        )

        # (b) BiGRU
        self.gru = nn.GRU(
            input_size=snapshot_dim,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=gru_dropout if gru_layers > 1 else 0.0,
        )
        gru_out = gru_hidden * 2  # bidirectional

        # (c) Transition detector (1D conv on GRU output)
        self.transition_head = nn.Sequential(
            nn.Conv1d(gru_out, gru_out // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(gru_out // 2, 1, kernel_size=1),
        )

        # (d) Pattern classifier (from onset-aligned hidden state)
        self.pattern_head = nn.Sequential(
            nn.Linear(gru_out, gru_out // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(gru_out // 2, n_patterns),
        )

        # aggregation projection
        self.agg_proj = nn.Sequential(
            nn.Linear(gru_out, gru_out),
            nn.LayerNorm(gru_out),
        )

        # cached for visualization
        self._last: Optional[Dict] = None

    def set_active_features(self, active_features: Sequence[str]) -> None:
        self.snapshot_enc.set_active_features(active_features)

    # -----------------------------------------------------------------
    # helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _pack_and_run_gru(
        gru: nn.GRU, x: torch.Tensor, lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Run GRU with packed sequences for variable lengths."""
        lengths_cpu = lengths.clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths_cpu, batch_first=True, enforce_sorted=False,
        )
        out_packed, _ = gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(
            out_packed, batch_first=True, total_length=x.size(1),
        )
        return out

    @staticmethod
    def _build_valid_mask(
        valid_counts: torch.Tensor, max_len: int,
    ) -> torch.Tensor:
        """[B] -> [B, max_len] bool mask."""
        idx = torch.arange(max_len, device=valid_counts.device)
        return idx.unsqueeze(0) < valid_counts.unsqueeze(1)

    @staticmethod
    def _run_gru_with_valid_mask(
        gru: nn.GRU,
        x: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run GRU only on exact valid slots, then scatter outputs back."""
        B, P, F_ = x.shape
        lengths = valid_mask.long().sum(dim=1)
        max_len = int(lengths.max().item()) if lengths.numel() else 0
        out_dim = gru.hidden_size * (2 if gru.bidirectional else 1)
        out = x.new_zeros(B, P, out_dim)
        if max_len <= 0:
            return out

        compact = x.new_zeros(B, max_len, F_)
        valid_indices = []
        for b in range(B):
            idx = valid_mask[b].nonzero(as_tuple=False).flatten()
            valid_indices.append(idx)
            n = int(idx.numel())
            if n > 0:
                compact[b, :n] = x[b, idx]

        packed = nn.utils.rnn.pack_padded_sequence(
            compact,
            lengths.clamp(min=1).cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_out, _ = gru(packed)
        compact_out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out,
            batch_first=True,
            total_length=max_len,
        )

        for b, idx in enumerate(valid_indices):
            n = int(idx.numel())
            if n > 0:
                out[b, idx] = compact_out[b, :n]
        return out

    # -----------------------------------------------------------------
    # forward
    # -----------------------------------------------------------------

    def forward(
        self,
        brain_networks: torch.Tensor,
        valid_patch_counts: torch.Tensor,
        seizure_relative_time: torch.Tensor,
        valid_patch_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args
        ----
        brain_networks       : [B, P, 22, 22, 4]
        valid_patch_counts   : [B]  (long), used when valid_patch_mask is absent
        seizure_relative_time: [B, P]
        valid_patch_mask     : optional [B, P] exact valid slot mask

        Returns
        -------
        dict with keys:
          network_features : [B, 256]
          transition_probs : [B, P]
          pattern_logits   : [B, 3]
          evolution_hidden : [B, P, 256]
        """
        B, P = brain_networks.shape[:2]
        dev = brain_networks.device
        if valid_patch_mask is not None:
            valid_mask = valid_patch_mask.to(device=dev, dtype=torch.bool)
            valid_patch_counts = valid_mask.long().sum(dim=1)
        else:
            valid_mask = self._build_valid_mask(valid_patch_counts, P)  # [B, P]

        # ── (a) Multi-branch snapshot encoding ──
        flat_nets = brain_networks.reshape(B * P, *brain_networks.shape[2:])
        if self.use_checkpoint and self.training:
            snap, branch_wts = checkpoint(
                self.snapshot_enc, flat_nets, use_reentrant=False,
            )
        else:
            snap, branch_wts = self.snapshot_enc(flat_nets)  # [B*P,128], [B*P,4]
        snap = snap.reshape(B, P, -1)                         # [B, P, 128]
        branch_wts = branch_wts.reshape(B, P, -1)             # [B, P, 4]

        # add time embedding
        t_emb = self.time_embed(
            seizure_relative_time.unsqueeze(-1)             # [B, P, 1]
        )                                                    # [B, P, 128]
        snap = snap + t_emb

        # zero out invalid patches
        snap = snap * valid_mask.unsqueeze(-1).float()

        # ── (b) BiGRU ──
        if valid_patch_mask is not None:
            gru_out = self._run_gru_with_valid_mask(
                self.gru, snap, valid_mask,
            )
        else:
            gru_out = self._pack_and_run_gru(
                self.gru, snap, valid_patch_counts,
            )                                                # [B, P, 256]
        gru_out = gru_out * valid_mask.unsqueeze(-1).float()

        # ── (c) Transition detection ──
        # restrict to window
        tw_lo, tw_hi = self.transition_window
        in_window = (
            (seizure_relative_time >= tw_lo)
            & (seizure_relative_time <= tw_hi)
            & valid_mask
        )  # [B, P]

        # 1D conv expects [B, C, P]
        tr_in = gru_out.permute(0, 2, 1)                    # [B, 256, P]
        tr_logits = self.transition_head(tr_in).squeeze(1)   # [B, P]
        # mask: only score patches inside window
        # use a safe minimum value for fp16
        min_val = torch.finfo(tr_logits.dtype).min
        tr_logits = tr_logits.masked_fill(~in_window, min_val)
        transition_probs = torch.sigmoid(tr_logits)
        transition_probs = transition_probs * valid_mask.float()

        # ── (d) Pattern classification ──
        # Use hidden state at the onset boundary (patch index closest to t=0)
        onset_dist = seizure_relative_time.abs()
        max_val = torch.finfo(onset_dist.dtype).max / 2  # safe maximum for any dtype
        onset_dist = onset_dist.masked_fill(~valid_mask, max_val)
        onset_idx = onset_dist.argmin(dim=1)                 # [B]
        onset_h = gru_out[
            torch.arange(B, device=dev), onset_idx
        ]                                                    # [B, 256]
        pattern_logits = self.pattern_head(onset_h)          # [B, 3]

        # ── Aggregated features ──
        # attention-weighted mean over valid patches
        attn_w = transition_probs / transition_probs.sum(
            dim=1, keepdim=True
        ).clamp(min=1e-8)                                    # [B, P]
        agg = (gru_out * attn_w.unsqueeze(-1)).sum(dim=1)   # [B, 256]
        network_features = self.agg_proj(agg)                # [B, 256]

        out = {
            'network_features': network_features,
            'transition_probs': transition_probs,
            'transition_logits': tr_logits,
            'pattern_logits':   pattern_logits,
            'evolution_hidden': gru_out,
            'branch_weights':   branch_wts,
        }
        self._last = {k: v.detach() for k, v in out.items()}
        self._last['seizure_relative_time'] = seizure_relative_time.detach()
        self._last['valid_mask'] = valid_mask.detach()
        return out

    # -----------------------------------------------------------------
    # Auxiliary targets
    # -----------------------------------------------------------------

    @staticmethod
    def compute_auxiliary_targets(
        seizure_relative_time: torch.Tensor,
        valid_mask: torch.Tensor,
        onset_half_width: float = 0.5,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate supervision targets from relative timestamps.

        Returns
        -------
        transition_targets : [B, P]  1 within +-onset_half_width of t=0
        pattern_targets    : [B]     0=baseline, 1=onset, 2=spread
        """
        # Transition: patches within +-0.5s of seizure onset
        tt = (seizure_relative_time.abs() <= onset_half_width).float()
        tt = tt * valid_mask.float()

        # Pattern: determine dominant phase from valid relative times only.
        med_values = []
        for b in range(seizure_relative_time.size(0)):
            vals = seizure_relative_time[b][valid_mask[b]]
            if vals.numel() == 0:
                vals = seizure_relative_time[b]
            med_values.append(vals.median())
        med = torch.stack(med_values, dim=0)                  # [B]
        pattern = torch.where(
            med < -1.0,
            torch.zeros_like(med).long(),                     # baseline
            torch.where(
                med < 1.0,
                torch.ones_like(med).long(),                  # onset
                torch.full_like(med, 2).long(),               # spread
            ),
        )
        return {'transition_targets': tt, 'pattern_targets': pattern}

    # -----------------------------------------------------------------
    # Visualization
    # -----------------------------------------------------------------

    def plot_evolution_curve(
        self, batch_idx: int = 0, save_path: Optional[str] = None,
    ):
        """Plot hidden-state norm + transition probs over time."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib required")
        d = self._last
        if d is None:
            raise RuntimeError("Call forward() first")

        t = d['seizure_relative_time'][batch_idx].cpu()
        vm = d['valid_mask'][batch_idx].cpu()
        t_v = t[vm].numpy()
        h = d['evolution_hidden'][batch_idx][vm].cpu()
        h_norm = h.norm(dim=-1).numpy()
        tp = d['transition_probs'][batch_idx][vm].cpu().numpy()

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
        ax1.plot(t_v, h_norm, 'o-', markersize=4)
        ax1.axvline(0, color='r', ls='--', label='onset')
        ax1.set_ylabel('Hidden norm')
        ax1.legend()
        ax2.bar(t_v, tp, width=0.4, alpha=0.7)
        ax2.set_ylabel('Transition prob')
        ax2.set_xlabel('Time relative to onset (s)')
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        return fig

    def highlight_transition_patches(
        self, batch_idx: int = 0, top_k: int = 3,
    ) -> torch.Tensor:
        """Return indices of top-k transition patches."""
        d = self._last
        tp = d['transition_probs'][batch_idx]
        vm = d['valid_mask'][batch_idx]
        tp = tp.masked_fill(~vm, -1)
        return tp.topk(top_k).indices

    def compare_patterns(self, pattern_idx: int = 1):
        """Print pattern logits breakdown (stub for extension)."""
        d = self._last
        logits = d['pattern_logits']
        probs = torch.softmax(logits, dim=-1)
        labels = ['baseline', 'onset', 'spread']
        print(f"Pattern distribution across batch (focus={labels[pattern_idx]}):")
        for i in range(probs.shape[0]):
            parts = ' | '.join(f'{labels[j]}={probs[i,j]:.3f}' for j in range(3))
            print(f"  sample {i}: {parts}")

    # -----------------------------------------------------------------
    # Branch interpretability
    # -----------------------------------------------------------------

    def visualize_branch_importance(
        self, batch_idx: int = 0, patch_idx: int = 0,
        save_path: Optional[str] = None,
    ):
        """Bar chart of 4-feature branch weights for one patch."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib required")
        d = self._last
        if d is None:
            raise RuntimeError("Call forward() first")
        w = d['branch_weights'][batch_idx, patch_idx].cpu().numpy()
        t = d['seizure_relative_time'][batch_idx, patch_idx].item()
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.bar(FEATURE_NAMES, w, color=['#e74c3c', '#3498db', '#2ecc71', '#f39c12'])
        ax.set_ylabel('Attention weight')
        ax.set_title(f'Branch importance  (sample {batch_idx}, patch {patch_idx}, t={t:+.1f}s)')
        ax.set_ylim(0, 1)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        return fig

    def plot_feature_contribution(
        self, save_path: Optional[str] = None,
    ):
        """Global feature importance across all valid patches in the last batch."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib required")
        d = self._last
        if d is None:
            raise RuntimeError("Call forward() first")
        wts = d['branch_weights']         # [B, P, 4]
        vm = d['valid_mask']              # [B, P]
        valid_wts = wts[vm]               # [n_valid, 4]
        mean_w = valid_wts.mean(dim=0).cpu().numpy()
        std_w = valid_wts.std(dim=0).cpu().numpy()

        fig, ax = plt.subplots(figsize=(5, 3))
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
        ax.bar(FEATURE_NAMES, mean_w, yerr=std_w, capsize=5, color=colors)
        ax.set_ylabel('Mean attention weight')
        ax.set_title('Global feature contribution')
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        return fig

    def extra_repr(self) -> str:
        return (
            f"snapshot={self.snapshot_dim}, gru_h={self.gru_hidden}x2, "
            f"patterns={self.n_patterns}, "
            f"window={self.transition_window}"
        )


# =====================================================================
# Self-test
# =====================================================================

def _test():
    torch.manual_seed(42)
    B, P, C, NF = 4, 20, 22, 4

    nets = torch.randn(B, P, C, C, NF).abs()   # connectivity >= 0
    counts = torch.tensor([20, 16, 14, 20])
    rel_t = torch.linspace(-4, 5.5, P).unsqueeze(0).expand(B, -1)

    model = DynamicNetworkEvolutionModel(
        n_channels=C, n_net_features=NF, max_patches=P,
    )
    print(model)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    out = model(nets, counts, rel_t)

    assert out['network_features'].shape == (B, 256)
    assert out['transition_probs'].shape == (B, P)
    assert out['pattern_logits'].shape == (B, 3)
    assert out['evolution_hidden'].shape == (B, P, 256)
    assert out['branch_weights'].shape == (B, P, 4)

    # branch weights should sum to 1 per patch
    bw_sum = out['branch_weights'].sum(dim=-1)
    assert torch.allclose(bw_sum, torch.ones_like(bw_sum), atol=1e-5), \
        "Branch weights should sum to 1"
    print(f"branch_weights   : {list(out['branch_weights'].shape)}, "
          f"mean={out['branch_weights'].mean(0).mean(0).tolist()}")

    # transition probs should be zero for invalid patches
    for i in range(B):
        inv = counts[i].item()
        if inv < P:
            assert out['transition_probs'][i, inv:].abs().sum() == 0, \
                f"sample {i}: invalid patches should have 0 transition prob"

    print(f"network_features : {list(out['network_features'].shape)}")
    print(f"transition_probs : range=[{out['transition_probs'].min():.4f}, "
          f"{out['transition_probs'].max():.4f}]")
    print(f"pattern_logits   : {list(out['pattern_logits'].shape)}")

    # auxiliary targets
    vm = model._build_valid_mask(counts, P)
    aux = model.compute_auxiliary_targets(rel_t, vm)
    assert aux['transition_targets'].shape == (B, P)
    assert aux['pattern_targets'].shape == (B,)

    # gradient flow — include all heads in loss
    loss = (out['network_features'].sum()
            + out['pattern_logits'].sum()
            + out['transition_probs'].sum()
            + out['branch_weights'].sum())
    loss.backward()
    no_grad = [n for n, p in model.named_parameters()
               if p.requires_grad and (p.grad is None or p.grad.abs().sum() == 0)]
    if no_grad:
        print(f"WARNING: {len(no_grad)} params without gradient: {no_grad}")
    else:
        print("Gradient flow: OK (all params)")

    # highlight
    top_idx = model.highlight_transition_patches(0, top_k=3)
    print(f"Top-3 transition patches (sample 0): {top_idx.tolist()}")

    model.compare_patterns(1)

    print("\n[PASS] All tests passed!")


if __name__ == '__main__':
    _test()
