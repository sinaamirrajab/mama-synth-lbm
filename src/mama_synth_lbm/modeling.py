from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel, UNet2DModel
from transformers import CLIPTextModel, CLIPTokenizer

logger = logging.getLogger(__name__)


class ConstantTextConditionedUNet2DModel(nn.Module):
    """Expose an SD text-conditioned UNet through a simple image-conditioned API."""

    def __init__(
        self,
        unet: UNet2DConditionModel,
        encoder_hidden_states: torch.Tensor,
        tokenizer: CLIPTokenizer,
        text_encoder: CLIPTextModel,
    ) -> None:
        super().__init__()
        self.unet = unet
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.text_encoder.requires_grad_(False)
        self.text_encoder.eval()
        self.register_buffer("encoder_hidden_states", encoder_hidden_states.detach().float(), persistent=False)

    @property
    def config(self):
        return self.unet.config

    def forward(self, sample: torch.Tensor, timestep: torch.Tensor, return_dict: bool = True, **kwargs):
        encoder_hidden_states = self.encoder_hidden_states.to(device=sample.device, dtype=sample.dtype)
        if encoder_hidden_states.shape[0] != sample.shape[0]:
            encoder_hidden_states = encoder_hidden_states.expand(sample.shape[0], -1, -1)
        return self.unet(
            sample,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

    def save_pretrained(self, path: str | Path) -> None:
        self.unet.save_pretrained(path)

    def enable_gradient_checkpointing(self) -> None:
        self.unet.enable_gradient_checkpointing()


def normalize_variant(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return None if text.lower() in {"", "none", "null"} else text


def expand_unet_conv_in(unet: UNet2DConditionModel, in_channels: int) -> UNet2DConditionModel:
    if unet.config.in_channels == in_channels:
        return unet
    if unet.config.in_channels != 4:
        raise ValueError(f"Can only expand a pretrained 4-channel UNet, got {unet.config.in_channels}.")
    if in_channels < 4:
        raise ValueError(f"Cannot shrink pretrained UNet input to {in_channels} channels.")

    old_conv = unet.conv_in
    new_conv = nn.Conv2d(
        in_channels,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
    )
    with torch.no_grad():
        new_conv.weight.zero_()
        new_conv.weight[:, :4].copy_(old_conv.weight)
        if old_conv.bias is not None and new_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias)
    unet.register_to_config(in_channels=in_channels)
    unet.conv_in = new_conv
    return unet


def _constant_prompt_embeds(
    pretrained_model_name_or_path: str,
    *,
    revision: str | None,
    prompt: str,
) -> tuple[torch.Tensor, CLIPTokenizer, CLIPTextModel]:
    tokenizer = CLIPTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=revision,
    )
    text_encoder = CLIPTextModel.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    text_encoder.requires_grad_(False)
    text_encoder.eval()
    inputs = tokenizer(
        [prompt],
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        embeds = text_encoder(inputs.input_ids, return_dict=False)[0]
    return embeds, tokenizer, text_encoder


def build_pretrained_text_unet(
    pretrained_model_name_or_path: str,
    *,
    in_channels: int,
    revision: str | None = None,
    variant: str | None = None,
    prompt: str = "",
) -> ConstantTextConditionedUNet2DModel:
    unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="unet",
        revision=revision,
        variant=normalize_variant(variant),
    )
    unet = expand_unet_conv_in(unet, in_channels=in_channels)
    unet.register_to_config(mri_lbm_unet_init="pretrained_text", mri_lbm_constant_prompt=prompt)
    embeds, tokenizer, text_encoder = _constant_prompt_embeds(
        pretrained_model_name_or_path,
        revision=revision,
        prompt=prompt,
    )
    model = ConstantTextConditionedUNet2DModel(unet, embeds, tokenizer, text_encoder)
    model.float()
    return model


def build_random_unet(*, in_channels: int, sample_size: int = 64) -> UNet2DModel:
    return UNet2DModel(
        sample_size=sample_size,
        in_channels=in_channels,
        out_channels=4,
        layers_per_block=2,
        block_out_channels=(192, 384, 768, 768),
        down_block_types=("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
    )


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
