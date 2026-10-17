"""
Minimal, self-contained MAE ViT encoder for the SCN pretrained checkpoints.

The model is plain PyTorch (no timm dependency for the network itself), with
parameter names identical to the MAE / timm ViT so the official
`checkpoint-799.pth` files load with no key remapping.

Main entry points
-----------------
    encoder = load_mae_encoder(ckpt_path, size="B")       # frozen-ready encoder
    tokens  = encoder(images)                             # (B, 1 + 196, D)
    layers  = encoder(images, return_layers=[3, 7, 11])   # list of token tensors

    model   = MAEClassifier(encoder, num_classes=4)       # encoder + head
    logits  = model(images)
    z       = model.forward_latent(images)                # (B, D) pooled latent

The classifier reproduces the recipe in
SCN-SSL/downstream/mae/main_downstream_merged.py (`--global_pool`):
mean of patch tokens -> fresh LayerNorm (fc_norm) -> Linear head.
"""

import math
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms


# ---------------------------------------------------------------------------
# Normalisation statistics used for pretraining and fine-tuning
# ---------------------------------------------------------------------------
# Enhanced SCN (matches the *_enhanced checkpoints)
ENHANCED_MEAN = [0.817, 0.443, 0.119]
ENHANCED_STD = [0.205, 0.292, 0.174]

# Original SCN
ORIGINAL_MEAN = [0.788, 0.409, 0.108]
ORIGINAL_STD = [0.211, 0.279, 0.167]

MODEL_CONFIGS = {
    # size: (embed_dim, depth, num_heads)
    "B": dict(embed_dim=768, depth=12, num_heads=12),
    "L": dict(embed_dim=1024, depth=24, num_heads=16),
}


# ---------------------------------------------------------------------------
# ViT building blocks (names match timm so checkpoints load directly)
# ---------------------------------------------------------------------------
def drop_path(x, drop_prob=0.0, training=False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = x.new_empty(shape).bernoulli_(keep_prob)
    return x * mask / keep_prob


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)                      # (B, D, H/16, W/16)
        return x.flatten(2).transpose(1, 2)   # (B, N, D)


class Attention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class Mlp(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class MAEEncoder(nn.Module):
    """ViT encoder of MAE (the decoder is not needed downstream)."""

    def __init__(self, img_size=224, patch_size=16, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4.0, drop_path_rate=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        self.patch_embed = PatchEmbed(img_size, patch_size, 3, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim))
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [Block(embed_dim, num_heads, mlp_ratio, dpr[i]) for i in range(depth)]
        )
        # Pretrained final LayerNorm of the MAE encoder.
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, x, return_layers=None):
        """
        x: (B, 3, H, W) normalised images.

        Returns the token sequence of the last block, shape (B, 1 + N, D),
        where index 0 is the CLS token and 1: are the N = (H/16)*(W/16)
        patch tokens in row-major order. No final norm is applied, so you can
        choose `encoder.norm(tokens)` (pretrained) or your own pooling/norm.

        If `return_layers` is a list of block indices (0-based, negative
        allowed), a list with the tokens after each of those blocks is
        returned instead.
        """
        x = self.patch_embed(x)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed

        wanted = None
        if return_layers is not None:
            wanted = {i % self.depth for i in return_layers}
            outs = {}

        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if wanted is not None and i in wanted:
                outs[i] = x

        if wanted is not None:
            return [outs[i % self.depth] for i in return_layers]
        return x

    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def interpolate_pos_embed(pos_embed, new_grid):
    """Bicubic resize of (1, 1 + g*g, D) position embeddings to new_grid."""
    cls_pe, patch_pe = pos_embed[:, :1], pos_embed[:, 1:]
    old_grid = int(math.sqrt(patch_pe.shape[1]))
    if old_grid == new_grid:
        return pos_embed
    D = patch_pe.shape[-1]
    patch_pe = patch_pe.reshape(1, old_grid, old_grid, D).permute(0, 3, 1, 2)
    patch_pe = F.interpolate(patch_pe, size=(new_grid, new_grid), mode="bicubic", align_corners=False)
    patch_pe = patch_pe.permute(0, 2, 3, 1).reshape(1, new_grid * new_grid, D)
    return torch.cat([cls_pe, patch_pe], dim=1)


def load_mae_encoder(ckpt_path, size="B", img_size=224, drop_path_rate=0.0, verbose=True):
    """
    Build an MAEEncoder and load the encoder weights from an MAE pretraining
    checkpoint (full checkpoint with decoder/optimizer, or an encoder-only
    file produced by `export_encoder_weights`).
    """
    cfg = MODEL_CONFIGS[size.upper()]
    encoder = MAEEncoder(img_size=img_size, drop_path_rate=drop_path_rate, **cfg)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt)

    encoder_keys = set(encoder.state_dict().keys())
    state = OrderedDict(
        (k[len("module."):] if k.startswith("module.") else k, v) for k, v in state.items()
    )
    state = OrderedDict((k, v) for k, v in state.items() if k in encoder_keys)
    state["pos_embed"] = interpolate_pos_embed(state["pos_embed"], encoder.patch_embed.grid_size)

    msg = encoder.load_state_dict(state, strict=True)
    if verbose:
        n = sum(p.numel() for p in encoder.parameters()) / 1e6
        epoch = ckpt.get("epoch", "?") if isinstance(ckpt, dict) else "?"
        print(f"Loaded MAE ViT-{size.upper()} encoder ({n:.1f}M params, epoch {epoch}) "
              f"from {ckpt_path}: {msg}")
    return encoder


def export_encoder_weights(ckpt_path, out_path):
    """Save an encoder-only copy (~4x smaller, no decoder/optimizer state)."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt)
    keep = {k: v for k, v in state.items()
            if not k.startswith("decoder") and k != "mask_token"}
    torch.save({"model": keep, "epoch": ckpt.get("epoch")}, out_path)
    print(f"Saved {len(keep)} encoder tensors to {out_path}")


# ---------------------------------------------------------------------------
# Classification model (encoder + head)
# ---------------------------------------------------------------------------
class MAEClassifier(nn.Module):
    """
    encoder -> pooling -> [latent_module] -> head

    pool="avg": mean of patch tokens + fresh LayerNorm (fc_norm). This is the
                `--global_pool` setting used in main_downstream_merged.py.
    pool="cls": CLS token after the pretrained encoder.norm.

    `latent_module` is an optional nn.Module applied to the pooled latent
    z (B, D) before the head -- a convenient place to plug in your own
    latent-space tricks. It must keep the output shape (B, D), or pass
    `latent_dim` to size the head accordingly.
    """

    def __init__(self, encoder, num_classes, pool="avg", latent_module=None, latent_dim=None):
        super().__init__()
        assert pool in ("avg", "cls")
        self.encoder = encoder
        self.pool = pool
        D = encoder.embed_dim
        self.fc_norm = nn.LayerNorm(D, eps=1e-6) if pool == "avg" else nn.Identity()
        self.latent_module = latent_module if latent_module is not None else nn.Identity()
        self.head = nn.Linear(latent_dim or D, num_classes)
        nn.init.trunc_normal_(self.head.weight, std=2e-5)
        nn.init.zeros_(self.head.bias)

    def pool_tokens(self, tokens):
        if self.pool == "avg":
            return self.fc_norm(tokens[:, 1:].mean(dim=1))
        return self.encoder.norm(tokens)[:, 0]

    def forward_latent(self, x):
        """Pooled latent representation z (B, D), before latent_module."""
        return self.pool_tokens(self.encoder(x))

    def forward(self, x):
        z = self.latent_module(self.forward_latent(x))
        return self.head(z)

    def no_weight_decay(self):
        return {"encoder." + k for k in self.encoder.no_weight_decay()}


def freeze_encoder(model):
    """Freeze the encoder only; fc_norm, latent_module and head stay trainable."""
    for p in model.encoder.parameters():
        p.requires_grad = False


# ---------------------------------------------------------------------------
# Optimiser helpers (layer-wise lr decay, as in MAE / BEiT fine-tuning)
# ---------------------------------------------------------------------------
def param_groups_lrd(model, weight_decay=0.05, layer_decay=0.75):
    """
    Parameter groups with layer-wise learning-rate decay. Each group carries
    an `lr_scale` that `adjust_learning_rate` multiplies into the lr.
    Layer 0 = patch_embed/cls_token/pos_embed, layer i+1 = block i,
    last layer = everything after the encoder blocks (norms, head, ...).
    """
    num_layers = model.encoder.depth + 1
    scales = [layer_decay ** (num_layers - i) for i in range(num_layers + 1)]
    no_wd = model.no_weight_decay()
    groups = {}

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 1 or name in no_wd:
            g_decay, this_wd = "no_decay", 0.0
        else:
            g_decay, this_wd = "decay", weight_decay

        if name.startswith(("encoder.cls_token", "encoder.pos_embed", "encoder.patch_embed")):
            layer_id = 0
        elif name.startswith("encoder.blocks."):
            layer_id = int(name.split(".")[2]) + 1
        else:
            layer_id = num_layers

        key = f"layer_{layer_id}_{g_decay}"
        if key not in groups:
            groups[key] = {"lr_scale": scales[layer_id], "weight_decay": this_wd, "params": []}
        groups[key]["params"].append(p)
    return list(groups.values())


def adjust_learning_rate(optimizer, epoch, lr, min_lr, epochs, warmup_epochs):
    """Linear warmup then half-cycle cosine; `epoch` may be fractional."""
    if epoch < warmup_epochs:
        cur = lr * epoch / warmup_epochs
    else:
        t = (epoch - warmup_epochs) / max(1e-8, epochs - warmup_epochs)
        cur = min_lr + (lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * t))
    for g in optimizer.param_groups:
        g["lr"] = cur * g.get("lr_scale", 1.0)
    return cur


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def build_transform(is_train, input_size=224, mean=ENHANCED_MEAN, std=ENHANCED_STD,
                    aa="rand-m9-mstd0.5-inc1", reprob=0.25):
    """
    Same transforms as SCN-SSL/downstream/mae/util/datasets.py.
    Train: timm RandAugment + RandomResizedCrop + flip + RandomErasing.
    Eval:  resize to input/0.875, centre crop.
    """
    if is_train:
        from timm.data import create_transform
        return create_transform(
            input_size=input_size, is_training=True, color_jitter=None,
            auto_augment=aa, interpolation="bicubic",
            re_prob=reprob, re_mode="pixel", re_count=1, mean=mean, std=std,
        )
    crop_pct = 224 / 256 if input_size <= 224 else 1.0
    return transforms.Compose([
        transforms.Resize(int(input_size / crop_pct), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def build_dataset(data_path, split, transform):
    """ImageFolder over <data_path>/<split>/<class>/*.png"""
    import os
    return datasets.ImageFolder(os.path.join(data_path, split), transform=transform)


def make_balanced_sampler(dataset):
    """Inverse-class-frequency sampler (`--train-sampling class_balanced`)."""
    targets = np.asarray(dataset.targets)
    weights = 1.0 / np.bincount(targets)[targets]
    return torch.utils.data.WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double), num_samples=len(targets), replacement=True
    )


# ---------------------------------------------------------------------------
# Metrics (same definitions as engine_finetune.py)
# ---------------------------------------------------------------------------
def classification_metrics(y_true, probs):
    """
    y_true: (N,) int labels, probs: (N, C) class probabilities.
    `score` = (macro-F1 + macro-AUROC + kappa) / 3 is the model-selection
    criterion used on the validation set in main_downstream_merged.py.
    """
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score, cohen_kappa_score,
                                 f1_score, roc_auc_score)
    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs, dtype=np.float64)
    C = probs.shape[1]
    y_pred = probs.argmax(1)
    labels = np.arange(C)
    try:
        if C == 2:
            auc = roc_auc_score(y_true, probs[:, 1])
        else:
            auc = roc_auc_score(y_true, probs / probs.sum(1, keepdims=True),
                                multi_class="ovr", average="macro", labels=labels)
    except ValueError:
        auc = float("nan")
    m = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "roc_auc": auc,
        "kappa": cohen_kappa_score(y_true, y_pred, labels=labels),
    }
    m["score"] = (m["f1"] + m["roc_auc"] + m["kappa"]) / 3
    return m
