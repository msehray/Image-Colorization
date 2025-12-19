# ----------------------------- inference_core.py ----------------------------------
# IMAGE-ONLY VERSION for ColorMNet (robusted)
# Removes video/mask dependencies and exposes a simple .colorize() method.

import torch
import torch.nn.functional as F


class InferenceCore:
    def __init__(self, network, config=None):
        """
        network : a ColorMNet instance (already moved to device)
        config  : optional dict of settings
        """
        self.network = network
        self.config = config if config is not None else {}
        # device derived from network parameters (assumes network has parameters)
        try:
            self.device = next(network.parameters()).device
        except StopIteration:
            # fallback to cpu
            self.device = torch.device("cpu")

    # ------------------------------------------------------------------
    #  IMAGE ONLY COLORIZATION CALL
    # ------------------------------------------------------------------
    def colorize(self, rgb_tensor, ref_tensor=None):
        """
        Pure image colorization.

        Inputs:
          - rgb_tensor: torch.Tensor, shape (C,H,W) or (B,C,H,W), dtype float, range [0,1]
          - ref_tensor: optional torch.Tensor, same shapes as rgb_tensor (C,H,W) or (B,C,H,W)

        Returns:
          - prob: torch.Tensor with AB-like logits. If input was single-image,
                  returns (C_out, H, W). If batched, returns (B, C_out, H, W).
        """
        # Normalize shapes: ensure we operate with batch dim
        single_input = False
        if not torch.is_tensor(rgb_tensor):
            raise TypeError("rgb_tensor must be a torch.Tensor")

        if rgb_tensor.dim() == 3:
            rgb = rgb_tensor.unsqueeze(0).to(self.device)  # [1,C,H,W]
            single_input = True
        elif rgb_tensor.dim() == 4:
            rgb = rgb_tensor.to(self.device)
        else:
            raise RuntimeError(f"Unsupported rgb_tensor shape: {tuple(rgb_tensor.shape)}")

        # prepare ref (optional)
        ref = None
        if ref_tensor is not None:
            if not torch.is_tensor(ref_tensor):
                raise TypeError("ref_tensor must be a torch.Tensor or None")
            if ref_tensor.dim() == 3:
                ref = ref_tensor.unsqueeze(0).to(self.device)
            elif ref_tensor.dim() == 4:
                ref = ref_tensor.to(self.device)
            else:
                raise RuntimeError(f"Unsupported ref_tensor shape: {tuple(ref_tensor.shape)}")

            # if batch sizes mismatch but ref is single, expand it
            if ref.size(0) != rgb.size(0) and ref.size(0) == 1 and rgb.size(0) > 1:
                ref = ref.repeat(rgb.size(0), 1, 1, 1)

        # Try exemplar-based API first if available
        # Note: this InferenceCore defines step_AnyExemplar and step for compatibility.
        if hasattr(self, "step_AnyExemplar"):
            try:
                # Build exemplar-like tensor: use L channel repeated to 3 if needed
                exemplar = None
                if ref is not None:
                    # if ref has >=1 channel, take first channel as L-like and repeat
                    exemplar = ref[:, :1, :, :].repeat(1, 3, 1, 1)
                else:
                    exemplar = rgb[:, :1, :, :].repeat(1, 3, 1, 1)

                # step_AnyExemplar expects per-sample (C,H,W) inputs in this implementation
                # We iterate per batch for safety (batch sizes are typically small)
                outputs = []
                for i in range(rgb.size(0)):
                    out = self.step_AnyExemplar(
                        rgb[i],
                        exemplar[i],
                        None,     # no long-term mask
                        None,     # no labels
                        end=True
                    )
                    outputs.append(out.unsqueeze(0))
                prob = torch.cat(outputs, dim=0)  # (B, C_out, H, W)
                if single_input:
                    return prob[0]
                return prob
            except Exception:
                # fall through to step() fallback
                pass

        # Fallback: call step() if present
        if hasattr(self, "step"):
            try:
                outputs = []
                for i in range(rgb.size(0)):
                    out = self.step(rgb[i], None, None, end=True)
                    outputs.append(out.unsqueeze(0))
                prob = torch.cat(outputs, dim=0)
                if single_input:
                    return prob[0]
                return prob
            except Exception as e:
                raise RuntimeError(f"InferenceCore.step() failed: {e}")

        raise RuntimeError(
            "InferenceCore: No usable colorization method found (step_AnyExemplar or step)."
        )

    # ------------------------------------------------------------------
    # THESE METHODS REMAIN FOR BACKWARD COMPATIBILITY WITH ORIGINAL CODE
    # (They forward to the underlying network and return per-sample outputs)
    # ------------------------------------------------------------------
    def step_AnyExemplar(self, rgb, exemplar, long_mask, labels, end=False):
        """
        Simplified compatibility wrapper. Expects rgb and exemplar as (C,H,W).
        Returns a tensor (C_out, H, W) for the single sample.
        """
        if not torch.is_tensor(rgb):
            raise TypeError("rgb must be a torch.Tensor")
        if not torch.is_tensor(exemplar):
            raise TypeError("exemplar must be a torch.Tensor")

        rgb = rgb.to(self.device)
        exemplar = exemplar.to(self.device)

        with torch.no_grad():
            out = self.network(rgb.unsqueeze(0), exemplar.unsqueeze(0))
        # expected out shape: (1, C_out, H, W)
        return out[0].detach()

    def step(self, rgb, mask, labels, end=False):
        """
        Generic single-sample forward wrapper. rgb is (C,H,W).
        """
        if not torch.is_tensor(rgb):
            raise TypeError("rgb must be a torch.Tensor")

        rgb = rgb.to(self.device)
        with torch.no_grad():
            out = self.network(rgb.unsqueeze(0))
        return out[0].detach()
