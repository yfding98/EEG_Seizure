#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BrainNetworkContrastivePretrainer -- 基于脑网络的对比学习预训练

对比策略:
  1. 时间对比: seizure-pre vs seizure-post (neg=interictal)
  2. 空间对比: left-hemisphere vs right-hemisphere
  3. 多尺度对比: same-patch wPLI vs GC (neg=different patch)

数据增强:
  a) 时间抖动  b) 频段掩码  c) 边扰动  d) 噪声注入
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# Feature indices in last dim of brain_networks
FEAT_GC, FEAT_TE, FEAT_AEC, FEAT_WPLI = 0, 1, 2, 3

# Left / right hemisphere channel sets (0-indexed in 22-ch TCP)
LEFT_CHS  = [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18]     # FP1-F7 chain + left central
RIGHT_CHS = [4, 5, 6, 7, 12, 13, 14, 15, 19, 20, 21]    # FP2-F8 chain + right central


# =====================================================================
# Config
# =====================================================================

@dataclass
class PretrainConfig:
    n_channels: int = 22
    n_features: int = 4
    gcn_hidden: int = 64
    embed_dim: int = 128
    proj_hidden: int = 256
    temperature: float = 0.1
    queue_size: int = 1024
    # augmentation
    time_jitter_samples: int = 10     # ±50ms @ 200Hz
    edge_drop_rate: float = 0.05
    noise_snr_db: float = 20.0
    band_mask_prob: float = 0.2       # prob of masking one band
    # loss weights
    w_time: float = 1.0
    w_space: float = 1.0
    w_scale: float = 1.0
    # GRL
    grl_lambda: float = 0.1


# =====================================================================
# GCN Layer (lightweight)
# =====================================================================

class _GCN(nn.Module):
    def __init__(self, in_d: int, out_d: int):
        super().__init__()
        self.W = nn.Linear(in_d, out_d)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        deg = adj.sum(-1).clamp(min=1e-6).pow(-0.5)
        A = adj * deg.unsqueeze(-1) * deg.unsqueeze(-2)
        return self.W(A @ h)


# =====================================================================
# Gradient Reversal Layer
# =====================================================================

class _GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.clone()

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lam * grad, None


def grad_reverse(x: torch.Tensor, lam: float = 1.0) -> torch.Tensor:
    return _GRL.apply(x, lam)


# =====================================================================
# Network Encoder
# =====================================================================

class NetworkEncoder(nn.Module):
    """[*, 22, 22, 4] -> [*, embed_dim]  via 2-layer GCN + pool."""

    def __init__(self, n_ch: int = 22, n_feat: int = 4,
                 hidden: int = 64, embed: int = 128):
        super().__init__()
        node_in = n_ch * n_feat   # 88
        self.gcn1 = _GCN(node_in, hidden)
        self.gcn2 = _GCN(hidden, hidden)
        self.proj = nn.Linear(hidden, embed)
        self.norm = nn.LayerNorm(embed)

    def forward(self, nets: torch.Tensor) -> torch.Tensor:
        lead = nets.shape[:-3]
        C = nets.shape[-3]
        adj = nets[..., FEAT_WPLI]                           # [*, C, C]
        node = nets.reshape(*lead, C, -1)                    # [*, C, C*4]
        h = F.relu(self.gcn1(node, adj))
        h = F.relu(self.gcn2(h, adj))                        # [*, C, hidden]
        h = h.mean(dim=-2)                                    # [*, hidden]
        return self.norm(self.proj(h))                        # [*, embed]


# =====================================================================
# Projection Head
# =====================================================================

class ProjectionHead(nn.Module):
    """2-layer MLP (embed -> proj_hidden -> embed)."""

    def __init__(self, embed: int = 128, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, embed),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


# =====================================================================
# Data Augmentations
# =====================================================================

class BrainNetworkAugmentor(nn.Module):
    """Apply augmentations to brain_networks [*, 22, 22, 4]."""

    def __init__(self, cfg: PretrainConfig):
        super().__init__()
        self.cfg = cfg

    @torch.no_grad()
    def edge_perturbation(self, nets: torch.Tensor) -> torch.Tensor:
        """Randomly zero out `edge_drop_rate` fraction of edges."""
        mask = torch.rand_like(nets[..., 0]) > self.cfg.edge_drop_rate
        mask = mask.unsqueeze(-1)  # [*, C, C, 1]
        return nets * mask

    @torch.no_grad()
    def noise_injection(self, nets: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise at given SNR."""
        snr_linear = 10 ** (self.cfg.noise_snr_db / 20.0)
        signal_power = nets.pow(2).mean()
        noise_std = (signal_power.sqrt() / snr_linear).clamp(min=1e-8)
        return nets + torch.randn_like(nets) * noise_std

    @torch.no_grad()
    def band_mask(self, nets: torch.Tensor) -> torch.Tensor:
        """Randomly zero one feature channel (simulating band masking)."""
        if torch.rand(1).item() > self.cfg.band_mask_prob:
            return nets
        feat_idx = torch.randint(0, nets.shape[-1], (1,)).item()
        out = nets.clone()
        out[..., feat_idx] = 0
        return out

    @torch.no_grad()
    def time_jitter(self, nets: torch.Tensor) -> torch.Tensor:
        """Shift patches along patch dimension by random offset."""
        if nets.dim() < 4:
            return nets
        P = nets.shape[-4]  # n_patches dim
        shift = torch.randint(-min(self.cfg.time_jitter_samples, P // 4),
                              min(self.cfg.time_jitter_samples, P // 4) + 1,
                              (1,)).item()
        if shift == 0 or P <= 1:
            return nets
        return torch.roll(nets, shifts=shift, dims=-4)

    def augment(self, nets: torch.Tensor) -> torch.Tensor:
        """Apply all augmentations sequentially."""
        x = self.edge_perturbation(nets)
        x = self.noise_injection(x)
        x = self.band_mask(x)
        x = self.time_jitter(x)
        return x


# =====================================================================
# Domain Discriminator (for GRL)
# =====================================================================

class DomainHead(nn.Module):
    def __init__(self, embed: int = 128, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor, lam: float = 1.0) -> torch.Tensor:
        return self.net(grad_reverse(x, lam))


# =====================================================================
# Main Module
# =====================================================================

class BrainNetworkContrastivePretrainer(nn.Module):
    """
    对比学习预训练器

    三种对比策略 + InfoNCE loss + 负样本队列 + 数据增强 + GRL域适应

    Parameters: see PretrainConfig
    """

    def __init__(self, cfg: PretrainConfig = None):
        super().__init__()
        self.cfg = cfg or PretrainConfig()
        c = self.cfg

        # encoder + projection
        self.encoder = NetworkEncoder(
            n_ch=c.n_channels, n_feat=c.n_features,
            hidden=c.gcn_hidden, embed=c.embed_dim,
        )
        self.projector = ProjectionHead(c.embed_dim, c.proj_hidden)

        # augmentor
        self.augmentor = BrainNetworkAugmentor(c)

        # domain discriminator
        self.domain_head = DomainHead(c.embed_dim)

        # negative sample queue (MoCo-style)
        self.register_buffer('queue', F.normalize(
            torch.randn(c.queue_size, c.embed_dim), dim=-1,
        ))
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))

    # -----------------------------------------------------------------
    # Queue management
    # -----------------------------------------------------------------

    @torch.no_grad()
    def _enqueue(self, keys: torch.Tensor):
        """Add new embeddings to the queue (FIFO)."""
        B = keys.shape[0]
        ptr = int(self.queue_ptr.item())
        qs = self.cfg.queue_size
        if ptr + B <= qs:
            self.queue[ptr: ptr + B] = keys.detach()
        else:
            overflow = (ptr + B) - qs
            self.queue[ptr:] = keys[:B - overflow].detach()
            self.queue[:overflow] = keys[B - overflow:].detach()
        self.queue_ptr[0] = (ptr + B) % qs

    # -----------------------------------------------------------------
    # InfoNCE
    # -----------------------------------------------------------------

    @staticmethod
    def compute_infonce_loss(
        z_anchor: torch.Tensor,
        z_pos: torch.Tensor,
        z_neg: torch.Tensor,
        temperature: float = 0.1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        InfoNCE = -log( exp(sim(a,p)/t) / (exp(sim(a,p)/t) + sum exp(sim(a,n)/t)) )

        Args:
          z_anchor, z_pos: [B, D]  (L2-normalised)
          z_neg: [K, D] or [B, K, D]  negative pool
          temperature: scalar

        Returns: loss, mean_sim_pos, mean_sim_neg
        """
        # positive similarity
        sim_pos = (z_anchor * z_pos).sum(-1) / temperature       # [B]

        if z_neg.dim() == 2:
            # shared neg pool [K, D]
            sim_neg = z_anchor @ z_neg.T / temperature           # [B, K]
        else:
            # per-sample negs [B, K, D]
            sim_neg = torch.bmm(
                z_neg, z_anchor.unsqueeze(-1),
            ).squeeze(-1) / temperature                          # [B, K]

        # log-sum-exp
        logits = torch.cat([sim_pos.unsqueeze(-1), sim_neg], dim=-1)  # [B, 1+K]
        labels = torch.zeros(logits.size(0), dtype=torch.long,
                             device=logits.device)               # positive = index 0
        loss = F.cross_entropy(logits, labels)

        return loss, sim_pos.mean().detach(), sim_neg.mean().detach()

    # -----------------------------------------------------------------
    # Contrastive strategies (build pairs from input)
    # -----------------------------------------------------------------

    def _strategy_temporal(
        self, nets: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Temporal contrast: pre-onset patches vs post-onset patches.
        nets: [B, P, 22, 22, 4]
        Returns: z_pre [B, D], z_post [B, D]
        """
        P = nets.shape[1]
        mid = P // 2
        pre  = nets[:, :mid].mean(dim=1)       # [B, 22, 22, 4]
        post = nets[:, mid:].mean(dim=1)       # [B, 22, 22, 4]
        z_pre  = self.projector(self.encoder(self.augmentor.augment(pre)))
        z_post = self.projector(self.encoder(self.augmentor.augment(post)))
        return z_pre, z_post

    def _strategy_spatial(
        self, nets: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Spatial contrast: left-hemisphere vs right-hemisphere subgraph.
        nets: [B, P, 22, 22, 4]
        Returns: z_left [B, D], z_right [B, D]
        """
        # average over patches first
        avg = nets.mean(dim=1)                                # [B, 22, 22, 4]
        # extract subgraphs — keep full 22x22 but zero out cross-hemisphere
        left = avg.clone()
        left[:, RIGHT_CHS] = 0
        left[:, :, RIGHT_CHS] = 0
        right = avg.clone()
        right[:, LEFT_CHS] = 0
        right[:, :, LEFT_CHS] = 0
        z_l = self.projector(self.encoder(self.augmentor.augment(left)))
        z_r = self.projector(self.encoder(self.augmentor.augment(right)))
        return z_l, z_r

    def _strategy_scale(
        self, nets: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Multi-scale contrast: wPLI view vs GC view of the same patch.
        nets: [B, P, 22, 22, 4]
        Returns: z_wpli [B, D], z_gc [B, D]
        """
        avg = nets.mean(dim=1)                                # [B, 22, 22, 4]
        # wPLI-only view
        wpli_view = torch.zeros_like(avg)
        wpli_view[..., FEAT_WPLI] = avg[..., FEAT_WPLI]
        # GC-only view
        gc_view = torch.zeros_like(avg)
        gc_view[..., FEAT_GC] = avg[..., FEAT_GC]
        z_w = self.projector(self.encoder(self.augmentor.augment(wpli_view)))
        z_g = self.projector(self.encoder(self.augmentor.augment(gc_view)))
        return z_w, z_g

    # -----------------------------------------------------------------
    # forward
    # -----------------------------------------------------------------

    def forward(
        self,
        brain_networks_pos: torch.Tensor,
        brain_networks_neg: torch.Tensor,
        seizure_labels: Optional[torch.Tensor] = None,
        domain_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args
        ----
        brain_networks_pos : [B, P, 22, 22, 4]  positive samples
        brain_networks_neg : [B, P, 22, 22, 4]  negative samples
        seizure_labels     : [B]  0=interictal, 1=ictal
        domain_labels      : [B]  0=public, 1=private

        Returns: dict with embeddings, contrastive_loss, similarities
        """
        c = self.cfg

        # ── Encode negatives into queue ──
        with torch.no_grad():
            neg_avg = brain_networks_neg.mean(dim=1)          # [B, 22, 22, 4]
            z_neg_raw = self.encoder(neg_avg)                 # [B, D]
            z_neg_proj = F.normalize(z_neg_raw, dim=-1)
            self._enqueue(z_neg_proj)

        neg_pool = self.queue.clone().detach()                # [Q, D]

        # ── Strategy 1: Temporal ──
        z_pre, z_post = self._strategy_temporal(brain_networks_pos)
        loss_time, sp_t, sn_t = self.compute_infonce_loss(
            z_pre, z_post, neg_pool, c.temperature,
        )

        # ── Strategy 2: Spatial ──
        z_left, z_right = self._strategy_spatial(brain_networks_pos)
        loss_space, sp_s, sn_s = self.compute_infonce_loss(
            z_left, z_right, neg_pool, c.temperature,
        )

        # ── Strategy 3: Multi-scale ──
        z_wpli, z_gc = self._strategy_scale(brain_networks_pos)
        loss_scale, sp_sc, sn_sc = self.compute_infonce_loss(
            z_wpli, z_gc, neg_pool, c.temperature,
        )

        # ── Combined loss ──
        contrastive_loss = (
            c.w_time * loss_time
            + c.w_space * loss_space
            + c.w_scale * loss_scale
        )

        # ── Domain adversarial (optional) ──
        domain_loss = torch.tensor(0.0, device=brain_networks_pos.device)
        if domain_labels is not None:
            # use anchor embedding
            pos_emb = self.encoder(brain_networks_pos.mean(dim=1))
            domain_logits = self.domain_head(pos_emb, c.grl_lambda)
            domain_loss = F.binary_cross_entropy_with_logits(
                domain_logits.squeeze(-1), domain_labels.float(),
            )
            contrastive_loss = contrastive_loss + c.grl_lambda * domain_loss

        # ── Output embeddings (for downstream) ──
        with torch.no_grad():
            out_emb = self.encoder(brain_networks_pos.mean(dim=1))

        return {
            'embeddings': out_emb,
            'contrastive_loss': contrastive_loss,
            'loss_time': loss_time.detach(),
            'loss_space': loss_space.detach(),
            'loss_scale': loss_scale.detach(),
            'loss_domain': domain_loss.detach(),
            'positive_similarity': (sp_t + sp_s + sp_sc) / 3,
            'negative_similarity': (sn_t + sn_s + sn_sc) / 3,
        }

    # -----------------------------------------------------------------
    # Save / load pretrained encoder
    # -----------------------------------------------------------------

    def save_pretrained_encoder(self, path: str):
        """Save only the encoder weights (for downstream fine-tuning)."""
        torch.save({
            'encoder': self.encoder.state_dict(),
            'config': self.cfg,
        }, path)

    def load_pretrained_encoder(self, path: str, map_location='cpu'):
        ckpt = torch.load(path, map_location=map_location)
        self.encoder.load_state_dict(ckpt['encoder'])

    def extra_repr(self) -> str:
        return (
            f"embed={self.cfg.embed_dim}, queue={self.cfg.queue_size}, "
            f"temp={self.cfg.temperature}"
        )


# =====================================================================
# Self-test
# =====================================================================

def _test():
    torch.manual_seed(42)

    B, P, C, F = 4, 10, 22, 4
    cfg = PretrainConfig(
        n_channels=C, n_features=F, gcn_hidden=32,
        embed_dim=64, proj_hidden=128, queue_size=32,
    )

    model = BrainNetworkContrastivePretrainer(cfg)
    print(model)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    pos = torch.randn(B, P, C, C, F).abs()
    neg = torch.randn(B, P, C, C, F).abs()
    sz_labels = torch.tensor([1, 0, 1, 0])
    dom_labels = torch.tensor([0, 0, 1, 1], dtype=torch.float)

    # forward
    out = model(pos, neg, sz_labels, dom_labels)

    assert out['embeddings'].shape == (B, cfg.embed_dim)
    assert out['contrastive_loss'].dim() == 0
    print(f"embeddings        : {list(out['embeddings'].shape)}")
    print(f"contrastive_loss  : {out['contrastive_loss']:.4f}")
    print(f"  loss_time       : {out['loss_time']:.4f}")
    print(f"  loss_space      : {out['loss_space']:.4f}")
    print(f"  loss_scale      : {out['loss_scale']:.4f}")
    print(f"  loss_domain     : {out['loss_domain']:.4f}")
    print(f"positive_similarity: {out['positive_similarity']:.4f}")
    print(f"negative_similarity: {out['negative_similarity']:.4f}")

    # gradient
    out['contrastive_loss'].backward()
    n_grad = sum(1 for p in model.parameters()
                 if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0)
    n_req = sum(1 for p in model.parameters() if p.requires_grad)
    print(f"Gradient: {n_grad}/{n_req} params have grad")

    # queue should be partially filled
    assert model.queue_ptr.item() == B
    print(f"Queue pointer: {model.queue_ptr.item()}/{cfg.queue_size}")

    # second forward to test queue growth
    out2 = model(pos, neg)
    assert model.queue_ptr.item() == 2 * B

    # InfoNCE sanity: identical inputs should have high positive sim
    model.zero_grad()
    identical_pos = pos.clone()
    out3 = model(identical_pos, neg)
    # positive sim should > negative sim after some training, but at init
    # just check shapes work
    print(f"Second-pass loss: {out2['contrastive_loss']:.4f}")

    # save / load encoder
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        tmp_path = f.name
    model.save_pretrained_encoder(tmp_path)
    model2 = BrainNetworkContrastivePretrainer(cfg)
    model2.load_pretrained_encoder(tmp_path)
    # check weights match
    for (n1, p1), (n2, p2) in zip(
        model.encoder.state_dict().items(),
        model2.encoder.state_dict().items(),
    ):
        assert torch.equal(p1, p2), f"Mismatch in {n1}"
    os.unlink(tmp_path)
    print("Encoder save/load: OK")

    print("\n[PASS] All tests passed!")


if __name__ == '__main__':
    _test()
