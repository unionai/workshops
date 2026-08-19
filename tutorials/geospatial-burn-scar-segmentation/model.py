"""
Prithvi-EO-2.0 encoder + a lightweight segmentation decoder.

Prithvi ships as a raw checkpoint whose "official" loader is TerraTorch — a 68-dependency
tree (geopandas, lightning, torchgeo, diffusers, pycocotools) that is a real failure
surface in a tutorial people are meant to `pip install -r` and run. Instead we vendor IBM's
own `prithvi_mae.py` (Apache-2.0, unmodified) and hang our own decoder off it, which keeps
the dependency list at torch + timm + einops + numpy.

The encoder is a ViT-L: 24 blocks, embed_dim 1024, 3D patch embedding of (1, 16, 16). At
512x512 with num_frames=1 that gives a 32x32 token grid. We tap four blocks spread through
the depth, reshape them to spatial feature maps, fuse, and upsample back to full
resolution.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from prithvi_mae import PrithviViT

# Prithvi-EO-2.0-300M pretraining statistics, from the model card's config.json.
# Bands are HLS S30: B02 (blue), B03 (green), B04 (red), B8A (narrow NIR),
# B11 (SWIR 1), B12 (SWIR 2). Units are surface reflectance scaled by 10,000.
PRITHVI_MEAN = [1087.0, 1342.0, 1433.0, 2734.0, 1958.0, 1363.0]
PRITHVI_STD = [2248.0, 2179.0, 2178.0, 1850.0, 1242.0, 1049.0]

# Which encoder blocks to tap for the decoder. Spread through the 24-block stack so the
# decoder sees both low-level texture and high-level semantics.
FEATURE_LAYERS = (5, 11, 17, 23)


class ConvBlock(nn.Module):
    """3x3 conv -> BN -> ReLU, optionally preceded by a 2x nearest-neighbour upsample."""

    def __init__(self, in_ch: int, out_ch: int, upsample: bool = True):
        super().__init__()
        self.upsample = upsample
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.block(x)


class PrithviSegmenter(nn.Module):
    """Prithvi ViT encoder + FPN-style decoder for binary burn-scar segmentation."""

    def __init__(
        self,
        img_size: int = 512,
        in_chans: int = 6,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        num_classes: int = 1,
        decoder_dim: int = 256,
        freeze_encoder: bool = True,
    ):
        super().__init__()
        self.img_size = img_size
        self.encoder = PrithviViT(
            img_size=img_size,
            patch_size=(1, 16, 16),
            num_frames=1,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=4.0,
        )

        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        # Fuse the four tapped feature maps, then climb 32 -> 512 in four 2x steps.
        self.fuse = nn.Sequential(
            nn.Conv2d(embed_dim * len(FEATURE_LAYERS), decoder_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU(inplace=True),
        )
        self.up = nn.Sequential(
            ConvBlock(decoder_dim, 128),
            ConvBlock(128, 64),
            ConvBlock(64, 32),
            ConvBlock(32, 32),
        )
        self.head = nn.Conv2d(32, num_classes, kernel_size=1)

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 6, H, W) normalized reflectance -> (B, num_classes, H, W) logits."""
        # The encoder is frozen in the default configuration, so skip building its graph.
        ctx = torch.no_grad() if self.freeze_encoder else torch.enable_grad()
        with ctx:
            feats = self.encoder.forward_features(x)
            feats = [feats[i] for i in FEATURE_LAYERS]
            spatial = self.encoder.prepare_features_for_image_model(feats)

        if self.freeze_encoder:
            spatial = [s.detach() for s in spatial]

        fused = self.fuse(torch.cat(spatial, dim=1))
        out = self.up(fused)
        logits = self.head(out)

        # Guard against any residual size drift from odd input sizes.
        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return logits


def load_pretrained_encoder(model: PrithviSegmenter, checkpoint_path: str) -> dict:
    """
    Load IBM's pretrained encoder weights into our segmenter.

    The released .pt is a plain state dict of the full MAE (encoder + MAE decoder). We keep
    only the encoder tensors and drop the MAE reconstruction head, which we don't use.
    Returns a small report dict so the pipeline can show what actually matched — silent
    key mismatches are the classic way a "fine-tuned" model turns out to be random weights.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt.get("state_dict", ckpt))

    encoder_state = {}
    for key, value in state.items():
        name = key
        for prefix in ("module.", "model."):
            if name.startswith(prefix):
                name = name[len(prefix):]
        if name.startswith("encoder."):
            name = name[len("encoder."):]
        elif name.startswith("decoder.") or name.startswith("mask_token"):
            continue  # MAE reconstruction head — not needed for segmentation
        encoder_state[name] = value

    target = model.encoder.state_dict()
    loadable = {
        k: v for k, v in encoder_state.items()
        if k in target and target[k].shape == v.shape
    }

    # Non-persistent, deterministically-regenerated buffers are expected to be absent or
    # shaped differently — they are not learned weights. pos_embed in particular is a
    # sin-cos buffer sized for the pretraining resolution; ours is built for img_size and
    # interpolated at runtime. Excluding these keeps the "kept random init" warning honest.
    learned = {name for name, _ in model.encoder.named_parameters()}
    missing = sorted((set(target) - set(loadable)) & learned)
    skipped = sorted(set(encoder_state) - set(loadable))

    model.encoder.load_state_dict(loadable, strict=False)

    return {
        "loaded": len(loadable),
        "in_checkpoint": len(encoder_state),
        "target_tensors": len(target),
        "skipped": skipped[:12],
        "missing": missing[:12],
        "n_skipped": len(skipped),
        "n_missing": len(missing),  # learned tensors only — buffers excluded
    }


MODEL_REPO = "ibm-nasa-geospatial/Prithvi-EO-2.0-300M"
MODEL_FILE = "Prithvi_EO_V2_300M.pt"


def split_state_dict(model: "PrithviSegmenter", freeze_encoder: bool) -> dict:
    """
    Weights worth persisting after training.

    When the encoder is frozen it is bit-identical to the public Prithvi checkpoint, so
    storing it produces a ~1.2 GB artifact of which ~1.2 GB is redundant. Saving only the
    decoder gives a ~6 MB artifact that every downstream tile task can pull cheaply; the
    encoder is rebuilt from the Hub, which is cached per container.
    """
    state = model.state_dict()
    if not freeze_encoder:
        return state
    return {k: v for k, v in state.items() if not k.startswith("encoder.")}


def load_segmenter(ckpt: dict, img_size: int) -> "PrithviSegmenter":
    """
    Rebuild a trained segmenter from a checkpoint produced by `split_state_dict`.

    Handles both the slim (decoder-only) and full checkpoints transparently.
    """
    from huggingface_hub import hf_hub_download

    freeze = ckpt.get("freeze_encoder", True)
    model = PrithviSegmenter(img_size=img_size, freeze_encoder=freeze)

    state = ckpt["state_dict"]
    has_encoder = any(k.startswith("encoder.") for k in state)
    if not has_encoder:
        # Slim checkpoint: restore the pretrained encoder from the Hub, then the decoder.
        encoder_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)
        load_pretrained_encoder(model, encoder_path)

    missing, unexpected = model.load_state_dict(state, strict=False)
    decoder_missing = [k for k in missing if not k.startswith("encoder.")]
    if decoder_missing:
        raise RuntimeError(f"Checkpoint is missing decoder tensors: {decoder_missing[:6]}")
    model.eval()
    return model


def normalize(scene: np.ndarray) -> np.ndarray:
    """
    (6, H, W) raw reflectance -> normalized float32 using Prithvi pretraining stats.

    HLS surface reflectance is distributed as int16 scaled by 10,000. Some derived copies
    ship as 0-1 floats instead, which would silently produce a near-constant input after
    normalization, so rescale those back up rather than trusting the dtype.
    """
    arr = scene.astype(np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size and np.nanmax(finite) <= 1.5:
        arr = arr * 10000.0

    mean = np.asarray(PRITHVI_MEAN, dtype=np.float32).reshape(-1, 1, 1)
    std = np.asarray(PRITHVI_STD, dtype=np.float32).reshape(-1, 1, 1)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return (arr - mean) / std
