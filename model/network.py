"""
IMAGE-ONLY ColorMNet
--------------------
This modified version removes all video / memory / multi-object logic
and exposes a simple forward(rgb, exemplar=None) API suitable for image
colorization.

All “mode router” behavior is removed:
    forward('encode_key', ...)
    forward('segment', ...)
    forward('read_memory', ...)
    
Now forward() simply colorizes a single image.
"""

import torch
import torch.nn as nn

from model.modules import KeyEncoder_DINOv2_v6, ValueEncoder, Decoder, KeyProjection
from model.attention import LocalGatedPropagation


class ColorMNet(nn.Module):
    def __init__(self, config, model_path=None, map_location=None):
        super().__init__()

        # Load hyperparameters (key_dim, value_dim, hidden_dim)
        model_weights = self.init_hyperparameters(config, model_path, map_location)

        # Core flags
        self.single_object = config.get("single_object", False)
        print(f"[ColorMNet] Single object mode: {self.single_object}")

        # Encoders
        self.key_encoder = KeyEncoder_DINOv2_v6()
        self.value_encoder = ValueEncoder(self.value_dim, self.hidden_dim, self.single_object)

        # Projection module
        self.key_proj = KeyProjection(1024, self.key_dim)

        # Short-term attention (used lightly for image-to-image propagation)
        self.short_term_attn = LocalGatedPropagation(
            d_qk=64,
            d_vu=512 * 2,
            num_head=1,
            dilation=1,
            use_linear=False,
            dropout=0,
            d_att=64,
            max_dis=7,
            expand_ratio=1
        )

        # Decoder (produces LAB-like logits)
        self.decoder = Decoder(self.value_dim, self.hidden_dim)

        # Load pretrained weights if provided
        if model_weights is not None:
            self.load_weights(model_weights, init_as_zero_if_needed=True)

    # -------------------------------------------------------------------------
    # SIMPLE FORWARD() — used by image-only inference
    # -------------------------------------------------------------------------
    def forward(self, rgb, exemplar=None):
        """
        rgb:       [B,C,H,W]
        exemplar:  [B,C,H,W] (optional reference image)

        Returns:
            prob: [B,2,H,W]  (AB channels or equivalent logits)
        """

        # ----- ENCODE RGB -----
        key, shrink, select, f16, f8, f4 = self.encode_key_internal(rgb)

        # ----- Build dummy masks (for compatibility with original architecture) -----
        # We simulate a pseudo-mask: everywhere=object
        B, _, H, W = rgb.shape
        masks = torch.ones((B, 1, H, W), device=rgb.device, dtype=torch.float32)

        # ----- Encode values -----
        g16, h16 = self.value_encoder(rgb, f16, None, masks)

        # Short-term propagation refinement (optional)
        # Use f16 as both query and memory
        memory = f16.unsqueeze(2)  # shape [B, C, T=1, H, W]
        query = f16[:, :self.key_dim, :, :]

        memory_val_short, _ = self.short_term_attn(
            query=query,
            key=memory[:, :self.key_dim, :, :, :],
            value=memory,
            shrinkage=None,
            size_2d=(H, W)
        )

        # Replace g16 with short-term refined version
        memory_val_short = memory_val_short.permute(1, 2, 0).view(B, 1, self.value_dim, H, W)
        g16 = memory_val_short

        # ----- Decode to AB logits -----
        # multi-scale encoder output:
        multi_scale_features = (f16, f8, f4)
        hidden_state, logits = self.decoder(*multi_scale_features, h16, g16, h_out=True)

        # Tanh maps to [-1,1] → original code expects this
        prob = torch.tanh(logits)

        # Return only logits/prob (AB-like channels)
        return prob


    # -------------------------------------------------------------------------
    # INTERNAL ENCODER WRAPPERS (simplified)
    # -------------------------------------------------------------------------
    def encode_key_internal(self, frame):
        """A simplified version of encode_key() for single images."""
        if frame.dim() != 4:
            raise RuntimeError("ColorMNet.encode_key_internal: expected [B,C,H,W]")

        f16, f8, f4 = self.key_encoder(frame)
        key, shrink, select = self.key_proj(f16, need_sk=True, need_ek=True)
        return key, shrink, select, f16, f8, f4

    # -------------------------------------------------------------------------
    # HYPERPARAMETERS
    # -------------------------------------------------------------------------
    def init_hyperparameters(self, config, model_path=None, map_location=None):
        if model_path is not None:
            model_weights = torch.load(model_path, map_location=map_location)
            self.key_dim = model_weights['key_proj.key_proj.weight'].shape[0]
            self.value_dim = model_weights['value_encoder.fuser.block2.conv2.weight'].shape[0]

            if 'decoder.hidden_update.transform.weight' not in model_weights:
                self.hidden_dim = 0
                self.disable_hidden = True
            else:
                self.hidden_dim = model_weights['decoder.hidden_update.transform.weight'].shape[0] // 3
                self.disable_hidden = False
        else:
            model_weights = None
            self.key_dim = config.get('key_dim', 64)
            self.value_dim = config.get('value_dim', 512)
            self.hidden_dim = config.get('hidden_dim', 64)
            self.disable_hidden = self.hidden_dim <= 0

        config['key_dim'] = self.key_dim
        config['value_dim'] = self.value_dim
        config['hidden_dim'] = self.hidden_dim
        return model_weights

    # -------------------------------------------------------------------------
    # LOAD WEIGHTS SAFELY
    # -------------------------------------------------------------------------
    def load_weights(self, src_dict, init_as_zero_if_needed=False):
        # Convert single-object weights to multi-object if needed
        for k in list(src_dict.keys()):
            if k == 'value_encoder.conv1.weight':
                if src_dict[k].shape[1] == 4:
                    print('[ColorMNet] Converting SO → MO weights.')

                    pads = torch.zeros((64, 1, 7, 7), device=src_dict[k].device)
                    if not init_as_zero_if_needed:
                        nn.init.orthogonal_(pads)
                    src_dict[k] = torch.cat([src_dict[k], pads], 1)

        self.load_state_dict(src_dict, strict=False)
