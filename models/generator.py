# Reference: https://github.com/huaidanquede/MUSE-Speech-Enhancement/tree/main/models/generator

import torch
import torch.nn as nn
import math
from torchvision.ops.deform_conv import DeformConv2d
from einops import rearrange
from .mamba_block import TFMambaBlock
from .codec_module import DenseEncoder, MagDecoder, PhaseDecoder

#####################################
class DWConv2d_BN(nn.Module):

    def __init__(
            self,
            in_ch,
            out_ch,
            kernel_size=1,
            stride=1,
            norm_layer=nn.BatchNorm2d,
            act_layer=nn.Hardswish,
            bn_weight_init=1,
            offset_clamp=(-1, 1)
    ):
        super().__init__()

        self.offset_clamp = offset_clamp
        self.offset_generator = nn.Sequential(nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=3,
                                                        stride=1, padding=1, bias=False, groups=in_ch),
                                              nn.Conv2d(in_channels=in_ch, out_channels=18,
                                                        kernel_size=1,
                                                        stride=1, padding=0, bias=False)
                                              )
        self.dcn = DeformConv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            groups=in_ch
        )
        self.pwconv = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False)
        self.act = act_layer() if act_layer is not None else nn.Identity()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        offset = self.offset_generator(x)

        if self.offset_clamp:
            offset = torch.clamp(offset, min=self.offset_clamp[0], max=self.offset_clamp[1])
        x = self.dcn(x, offset)

        x = self.pwconv(x)
        x = self.act(x)
        return x


class MB_Deform_Embedding(nn.Module):

    def __init__(self,
                 in_chans=3,
                 embed_dim=768,
                 patch_size=16,
                 stride=1,
                 act_layer=nn.Hardswish,
                 offset_clamp=(-1, 1)):
        super().__init__()

        self.patch_conv = DWConv2d_BN(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            act_layer=act_layer,
            offset_clamp=offset_clamp
        )

    def forward(self, x):
        """foward function"""
        x = self.patch_conv(x)

        return x


class Patch_Embed_stage(nn.Module):
    """Depthwise Convolutional Patch Embedding stage comprised of
    `DWCPatchEmbed` layers."""

    def __init__(self, in_chans, embed_dim, isPool=False, offset_clamp=(-1, 1)):
        super(Patch_Embed_stage, self).__init__()

        self.patch_embeds = MB_Deform_Embedding(
                in_chans=in_chans,
                embed_dim=embed_dim,
                patch_size=3,
                stride=1,
                offset_clamp=offset_clamp)

    def forward(self, x):
        """foward function"""

        att_inputs = self.patch_embeds(x)

        return att_inputs

#####################################
class Downsample(nn.Module):
    def __init__(self, input_feat, out_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(
            # dw
            nn.Conv2d(input_feat, input_feat, kernel_size=3, stride=1, padding=1, groups=input_feat, bias=False),
            # pw-linear
            nn.Conv2d(input_feat, out_feat // 4, 1, 1, 0, bias=False),
            nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, input_feat, out_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(
            # dw
            nn.Conv2d(input_feat, input_feat, kernel_size=3, stride=1, padding=1, groups=input_feat, bias=False),
            # pw-linear
            nn.Conv2d(input_feat, out_feat * 4, 1, 1, 0, bias=False),
            nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

class MambaSEUNet(nn.Module):
    """
    SEMamba model for speech enhancement using Mamba blocks.
    
    This model uses a dense encoder, multiple Mamba blocks, and separate magnitude
    and phase decoders to process noisy magnitude and phase inputs.
    """
    def __init__(self, cfg):
        """
        Initialize the SEMamba model.
        
        Args:
        - cfg: Configuration object containing model parameters.
        """
        super(MambaSEUNet, self).__init__()
        self.cfg = cfg
        self.num_tscblocks = cfg['model_cfg']['num_tfmamba'] if cfg['model_cfg']['num_tfmamba'] is not None else 4  # default tfmamba: 4

        self.dim = [cfg['model_cfg']['hid_feature'], cfg['model_cfg']['hid_feature'] * 2, cfg['model_cfg']['hid_feature'] * 3]
        dim = self.dim

        # Initialize dense encoder
        self.dense_encoder = DenseEncoder(cfg)

        # Initialize Mamba blocks
        self.patch_embed_encoder_level1 = Patch_Embed_stage(dim[0], dim[0])

        self.TSMamba1_encoder = nn.ModuleList([TFMambaBlock(cfg, dim[0]) for _ in range(self.num_tscblocks)])

        self.down1_2 = Downsample(dim[0], dim[1])

        self.patch_embed_encoder_level2 = Patch_Embed_stage(dim[1], dim[1])

        self.TSMamba2_encoder = nn.ModuleList([TFMambaBlock(cfg, dim[1]) for _ in range(self.num_tscblocks)])

        self.down2_3 = Downsample(dim[1], dim[2])

        self.patch_embed_middle = Patch_Embed_stage(dim[2], dim[2])

        self.TSMamba_middle = nn.ModuleList([TFMambaBlock(cfg, dim[2]) for _ in range(self.num_tscblocks)])

        ###########

        self.up3_2 = Upsample(int(dim[2]), dim[1])

        self.concat_level2 = nn.Sequential(
            nn.Conv2d(dim[1] * 2, dim[1], 1, 1, 0, bias=False),
        )

        self.patch_embed_decoder_level2 = Patch_Embed_stage(dim[1], dim[1])

        self.TSMamba2_decoder = nn.ModuleList([TFMambaBlock(cfg, dim[1]) for _ in range(self.num_tscblocks)])

        self.up2_1 = Upsample(int(dim[1]), dim[0])

        self.concat_level1 = nn.Sequential(
            nn.Conv2d(dim[0] * 2, dim[0], 1, 1, 0, bias=False),
        )

        self.patch_embed_decoder_level1 = Patch_Embed_stage(dim[0], dim[0])

        self.TSMamba1_decoder = nn.ModuleList([TFMambaBlock(cfg, dim[0]) for _ in range(self.num_tscblocks)])

        # 幅度
        self.mag_patch_embed_refinement = Patch_Embed_stage(dim[0], dim[0])

        self.mag_refinement = nn.ModuleList([TFMambaBlock(cfg, dim[0]) for _ in range(self.num_tscblocks)])

        self.mag_output = nn.Sequential(
            nn.Conv2d(dim[0], dim[0], kernel_size=3, stride=1, padding=1, bias=False),

        )

        # 相位
        self.pha_patch_embed_refinement = Patch_Embed_stage(dim[0], dim[0])

        self.pha_refinement = nn.ModuleList([TFMambaBlock(cfg, dim[0]) for _ in range(self.num_tscblocks)])

        self.pha_output = nn.Sequential(
            nn.Conv2d(dim[0], dim[0], kernel_size=3, stride=1, padding=1, bias=False),

        )

        # Initialize decoders
        self.mask_decoder = MagDecoder(cfg)
        self.phase_decoder = PhaseDecoder(cfg)

    def forward(self, noisy_mag, noisy_pha):
        """
        Forward pass for the SEMamba model.
        
        Args:
        - noisy_mag (torch.Tensor): Noisy magnitude input tensor [B, F, T].
        - noisy_pha (torch.Tensor): Noisy phase input tensor [B, F, T].
        
        Returns:
        - denoised_mag (torch.Tensor): Denoised magnitude tensor [B, F, T].
        - denoised_pha (torch.Tensor): Denoised phase tensor [B, F, T].
        - denoised_com (torch.Tensor): Denoised complex tensor [B, F, T, 2].
        """
        # Reshape inputs
        noisy_mag = rearrange(noisy_mag, 'b f t -> b t f').unsqueeze(1)  # [B, 1, T, F]
        noisy_pha = rearrange(noisy_pha, 'b f t -> b t f').unsqueeze(1)  # [B, 1, T, F]

        # Concatenate magnitude and phase inputs
        x = torch.cat((noisy_mag, noisy_pha), dim=1)  # [B, 2, T, F]

        # Encode input
        x1 = self.dense_encoder(x)

        # Apply U-Net Mamba blocks
        copy1 = x1
        x1 = self.patch_embed_encoder_level1(x1)
        for block in self.TSMamba1_encoder:
            x1 = block(x1)
        x1 = copy1 + x1

        x2 = self.down1_2(x1)

        copy2 = x2
        x2 = self.patch_embed_encoder_level2(x2)
        for block in self.TSMamba2_encoder:
            x2 = block(x2)
        x2 = copy2 + x2

        x3 = self.down2_3(x2)

        copy3 = x3
        x3 = self.patch_embed_middle(x3)
        for block in self.TSMamba_middle:
            x3 = block(x3)
        x3 = copy3 + x3

        y2 = self.up3_2(x3)
        y2 = torch.cat([y2, x2], 1)
        y2 = self.concat_level2(y2)

        copy_de2 = y2
        y2 = self.patch_embed_decoder_level2(y2)
        for block in self.TSMamba2_decoder:
            y2 = block(y2)
        y2 = copy_de2 + y2

        y1 = self.up2_1(y2)
        y1 = torch.cat([y1, x1], 1)
        y1 = self.concat_level1(y1)

        copy_de1 = y1
        y1 = self.patch_embed_decoder_level1(y1)
        for block in self.TSMamba1_decoder:
            y1 = block(y1)
        y1 = copy_de1 + y1

        mag_input = y1
        pha_input = y1

        # magnitude
        copy_mag = mag_input
        mag_input = self.mag_patch_embed_refinement(mag_input)
        for block in self.mag_refinement:
            mag_input = block(mag_input)
        mag = copy_mag + mag_input
        mag = self.mag_output(mag) + copy1

        # phase
        copy_pha = pha_input
        pha_input = self.pha_patch_embed_refinement(pha_input)
        for block in self.pha_refinement:
            pha_input = block(pha_input)
        pha = copy_pha + pha_input
        pha = self.pha_output(pha) + copy1

        # Decode magnitude and phase
        denoised_mag = rearrange(self.mask_decoder(mag) * noisy_mag, 'b c t f -> b f t c').squeeze(-1)
        denoised_pha = rearrange(self.phase_decoder(pha), 'b c t f -> b f t c').squeeze(-1)

        # Combine denoised magnitude and phase into a complex representation
        denoised_com = torch.stack(
            (denoised_mag * torch.cos(denoised_pha), denoised_mag * torch.sin(denoised_pha)),
            dim=-1
        )

        return denoised_mag, denoised_pha, denoised_com
