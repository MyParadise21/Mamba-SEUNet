import torch
import torch.nn as nn
from einops import rearrange
from .mamba_block import TFMambaBlock
from .codec_module import DenseEncoder, MagDecoder, PhaseDecoder

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

class SEMamba(nn.Module):
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
        super(SEMamba, self).__init__()
        self.cfg = cfg
        self.num_tscblocks = cfg['model_cfg']['num_tfmamba'] if cfg['model_cfg']['num_tfmamba'] is not None else 4  # default tfmamba: 4

        self.dim = [cfg['model_cfg']['hid_feature'], cfg['model_cfg']['hid_feature'] * 2, cfg['model_cfg']['hid_feature'] * 3]
        dim = self.dim

        # Initialize dense encoder
        self.dense_encoder = DenseEncoder(cfg)

        # Initialize Mamba blocks
        self.TSMamba1_encoder = nn.ModuleList([TFMambaBlock(cfg, dim[0], 0) for _ in range(self.num_tscblocks)])

        self.down1_2 = Downsample(dim[0], dim[1])

        self.TSMamba2_encoder = nn.ModuleList([TFMambaBlock(cfg, dim[1], 1) for _ in range(self.num_tscblocks)])

        self.down2_3 = Downsample(dim[1], dim[2])

        self.TSMamba_middle = nn.ModuleList([TFMambaBlock(cfg, dim[2], 2) for _ in range(self.num_tscblocks)])

        ###########

        self.up3_2 = Upsample(int(dim[2]), dim[1])

        self.concat_level2 = nn.Sequential(
            nn.Conv2d(dim[1] * 2, dim[1], 1, 1, 0, bias=False),
        )

        self.TSMamba2_decoder = nn.ModuleList([TFMambaBlock(cfg, dim[1], 1) for _ in range(self.num_tscblocks)])

        self.up2_1 = Upsample(int(dim[1]), dim[0])

        self.concat_level1 = nn.Sequential(
            nn.Conv2d(dim[0] * 2, dim[0], 1, 1, 0, bias=False),
        )

        self.TSMamba1_decoder = nn.ModuleList([TFMambaBlock(cfg, dim[0], 0) for _ in range(self.num_tscblocks)])

        # 幅度
        self.mag_refinement = nn.ModuleList([TFMambaBlock(cfg, dim[0], 0) for _ in range(self.num_tscblocks)])

        self.mag_output = nn.Sequential(
            nn.Conv2d(dim[0], dim[0], kernel_size=3, stride=1, padding=1, bias=False),

        )

        # 相位
        self.pha_refinement = nn.ModuleList([TFMambaBlock(cfg, dim[0], 0) for _ in range(self.num_tscblocks)])

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
        for block in self.TSMamba1_encoder:
            x1 = block(x1)
        x1 = copy1 + x1

        x2 = self.down1_2(x1)

        copy2 = x2
        for block in self.TSMamba2_encoder:
            x2 = block(x2)
        x2 = copy2 + x2

        x3 = self.down2_3(x2)

        copy3 = x3
        for block in self.TSMamba_middle:
            x3 = block(x3)
        x3 = copy3 + x3

        y2 = self.up3_2(x3)
        y2 = torch.cat([y2, x2], 1)
        y2 = self.concat_level2(y2)

        copy_de2 = y2
        for block in self.TSMamba2_decoder:
            y2 = block(y2)
        y2 = copy_de2 + y2

        y1 = self.up2_1(y2)
        y1 = torch.cat([y1, x1], 1)
        y1 = self.concat_level1(y1)

        copy_de1 = y1
        for block in self.TSMamba1_decoder:
            y1 = block(y1)
        y1 = copy_de1 + y1

        mag_input = y1
        pha_input = y1

        # magnitude
        copy_mag = mag_input
        for block in self.mag_refinement:
            mag_input = block(mag_input)
        mag = copy_mag + mag_input
        mag = self.mag_output(mag) + copy1

        # phase
        copy_pha = pha_input
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
