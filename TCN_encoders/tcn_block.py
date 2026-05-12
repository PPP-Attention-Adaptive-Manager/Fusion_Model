"""
TCN Building Blocks — CausalConv1d + TemporalBlock.

These are the shared primitives used by every modality's FixedTCNEncoder.
Nothing here is modality-specific. Nothing here is trainable once frozen.

CausalConv1d  — single dilated causal convolution (left-pad only)
TemporalBlock — two CausalConv1d layers + residual + LayerNorm + GELU + Dropout
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    """
    Dilated causal 1D convolution.

    Causality is enforced by left-padding only — output[t] depends solely
    on input[0..t]. No future leakage at any dilation level.

    Args:
        in_channels  : number of input channels
        out_channels : number of output channels
        kernel_size  : width of the conv kernel
        dilation     : gap between kernel taps (1 = standard conv)
    """

    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : int = 3,
        dilation    : int = 1,
    ) -> None:
        super().__init__()
        # left-pad exactly enough to keep output length == input length
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size = kernel_size,
            dilation    = dilation,
            padding     = 0,      # manual causal padding below
            bias        = False,  # LayerNorm in TemporalBlock handles bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        x = F.pad(x, (self.pad, 0))   # pad left only → causal
        return self.conv(x)            # (B, out_channels, T)


class TemporalBlock(nn.Module):
    """
    One residual block of the TCN stack.

    Structure per block:
        x → CausalConv → LayerNorm → GELU → Dropout
          → CausalConv → LayerNorm → GELU → Dropout
          → + residual (1×1 conv projection if channels differ)
          → output

    LayerNorm operates on the channel dim (C), not the time dim,
    which is more stable than BatchNorm for variable-length sequences.

    Args:
        in_channels  : input channel count
        out_channels : output channel count
        kernel_size  : kernel size passed to both CausalConv1d layers
        dilation     : dilation passed to both CausalConv1d layers
        dropout      : dropout probability (applied after each activation)
    """

    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : int   = 3,
        dilation    : int   = 1,
        dropout     : float = 0.1,
    ) -> None:
        super().__init__()

        self.conv1 = CausalConv1d(in_channels,  out_channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)

        # LayerNorm on channel dim: input is (B, C, T), norm over C
        self.norm1 = nn.LayerNorm(out_channels)
        self.norm2 = nn.LayerNorm(out_channels)

        self.drop  = nn.Dropout(dropout)

        # residual projection — only instantiated when channels change
        self.residual_proj = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_channels, T)
        residual = self.residual_proj(x)

        out = self.conv1(x)
        out = self.norm1(out.transpose(1, 2)).transpose(1, 2)  # norm over C
        out = F.gelu(out)
        out = self.drop(out)

        out = self.conv2(out)
        out = self.norm2(out.transpose(1, 2)).transpose(1, 2)
        out = F.gelu(out)
        out = self.drop(out)

        return out + residual   # (B, out_channels, T)