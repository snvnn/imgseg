import torch
from torch import nn
from typing import Optional

def get_norm_layer(num_channels, kind="bn"):
    if kind == "bn":
        return nn.BatchNorm2d(num_channels)
    elif kind == "gn":
        # 채널 수에 따라 그룹 수 자동 조절(8 또는 16 권장)
        groups = 16 if num_channels >= 32 else 8
        groups = min(groups, num_channels)
        return nn.GroupNorm(groups, num_channels)
    else:
        return nn.Identity()

class SEBlock(nn.Module):
    def __init__(self, ch, r=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(ch, ch // r, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch // r, ch, 1, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        w = self.fc(self.pool(x))
        return x * w

class ResidualConv(nn.Module):
    def __init__(self, in_ch, out_ch, norm="bn", se=False, drop=0.0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            get_norm_layer(out_ch, norm), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            get_norm_layer(out_ch, norm)
        )
        self.act = nn.ReLU(inplace=True)
        self.proj = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.se = SEBlock(out_ch) if se else nn.Identity()
        self.drop = nn.Dropout2d(p=drop) if drop > 0 else nn.Identity()
    def forward(self, x):
        res = self.proj(x)
        x = self.conv(x)
        x = self.se(x)
        x = self.drop(x)
        return self.act(x + res)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch, norm="bn", se=False, drop=0.0):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.block = ResidualConv(in_ch, out_ch, norm=norm, se=se, drop=drop)
    def forward(self, x):
        return self.block(self.pool(x))

class AttentionGate(nn.Module):
    """
    Skip connection에 사용하는 Attention Gate.
    g: decoder feature (업샘플된 feature)
    x: encoder에서 온 skip feature
    """
    def __init__(self, F_g, F_x, F_int, norm="bn"):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=False),
            get_norm_layer(F_int, norm),
            nn.ReLU(inplace=True)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_x, F_int, kernel_size=1, bias=False),
            get_norm_layer(F_int, norm),
            nn.ReLU(inplace=True)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=True),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # g: upsampled decoder feature, x: encoder skip feature
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        # 두 feature를 더한 뒤 비선형 변환 → 1채널 마스크
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)  # [B, 1, H, W]

        # skip feature에 마스크를 곱해 중요한 위치만 통과
        return x * psi

class Up(nn.Module):
    def __init__(self, in_ch, out_ch, norm="bn", se=False, drop=0.0):
        super().__init__()
        # 업샘플은 bilinear + 1x1로 깔끔하게
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_ch, out_ch, 1, bias=False)
        )
        # decoder feature(g)와 encoder skip(x)에 대한 attention gate
        self.attn = AttentionGate(F_g=out_ch, F_x=out_ch, F_int=out_ch // 2, norm=norm)

        # attention으로 필터링된 skip과 concat 후 residual block
        self.block = ResidualConv(out_ch*2, out_ch, norm=norm, se=se, drop=drop)

    def forward(self, x, skip):
        # decoder feature upsample
        x = self.up(x)

        # 크기 보정(odd 입력일 때)
        if x.shape[-1] != skip.shape[-1] or x.shape[-2] != skip.shape[-2]:
            x = nn.functional.pad(
                x,
                (0, skip.shape[-1] - x.shape[-1], 0, skip.shape[-2] - x.shape[-2])
            )

        # skip connection에 attention 적용
        skip = self.attn(x, skip)

        # concat 후 convolution block
        x = torch.cat([x, skip], dim=1)
        return self.block(x)

class ASPP(nn.Module):
    """가볍게 쓸 수 있는 ASPP(선택). 멀티스케일 컨텍스트 강화."""
    def __init__(self, in_ch, out_ch, rates=(1, 6, 12, 18), norm="bn"):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1 if r==1 else 3, padding=0 if r==1 else r, dilation=r, bias=False),
                get_norm_layer(out_ch, norm),
                nn.ReLU(inplace=True)
            ) for r in rates
        ])
        self.project = nn.Sequential(
            nn.Conv2d(out_ch*len(rates), out_ch, 1, bias=False),
            get_norm_layer(out_ch, norm),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        feats = [b(x) for b in self.branches]
        x = torch.cat(feats, dim=1)
        return self.project(x)

class ImprovedUNet(nn.Module):
    """
    - in_channel: 입력 채널(예: RGB=3)
    - out_channel: 클래스 수(배경 포함)
    - img_size: (H,W) 안 써도 OK, 필요 시 검증용으로 들고만 다님
    - norm: 'bn' 또는 'gn'
    - use_aspp: 마지막 bottleneck에 ASPP 사용 여부
    """
    def __init__(self, in_channel, out_channel, img_size: Optional[tuple]=None,
                 base_ch=32, norm="bn", se=True, drop=0.1, use_aspp=False):
        super().__init__()
        self.img_size = img_size

        # Encoder
        self.enc1 = ResidualConv(in_channel,  base_ch,      norm=norm, se=se, drop=0.0)
        self.enc2 = Down(base_ch,             base_ch*2,    norm=norm, se=se, drop=drop)
        self.enc3 = Down(base_ch*2,           base_ch*4,    norm=norm, se=se, drop=drop)
        self.enc4 = Down(base_ch*4,           base_ch*8,    norm=norm, se=se, drop=drop)

        # Bottleneck
        self.bottleneck = ResidualConv(base_ch*8, base_ch*16, norm=norm, se=se, drop=drop)
        self.aspp = ASPP(base_ch*16, base_ch*16, rates=(1, 6, 12, 18), norm=norm) if use_aspp else nn.Identity()

        # Decoder
        self.up3 = Up(base_ch*16, base_ch*8,  norm=norm, se=se, drop=drop)
        self.up2 = Up(base_ch*8,  base_ch*4,  norm=norm, se=se, drop=drop)
        self.up1 = Up(base_ch*4,  base_ch*2,  norm=norm, se=se, drop=drop)
        self.up0 = Up(base_ch*2,  base_ch,    norm=norm, se=se, drop=0.0)

        self.head = nn.Conv2d(base_ch, out_channel, kernel_size=1)

        # 간단한 Kaiming init
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # encoder
        e1 = self.enc1(x)      # H
        e2 = self.enc2(e1)     # H/2
        e3 = self.enc3(e2)     # H/4
        e4 = self.enc4(e3)     # H/8

        # bottleneck (+ optional ASPP)
        b  = self.bottleneck(e4)
        b  = self.aspp(b)

        # decoder with skips
        d3 = self.up3(b,  e4)  # H/8
        d2 = self.up2(d3, e3)  # H/4
        d1 = self.up1(d2, e2)  # H/2
        d0 = self.up0(d1, e1)  # H

        logits = self.head(d0) # [B, out_channel, H, W]
        return logits


# 조금 더 깊은 U-Net 구조 (DeepUNet)
class DeepUNet(nn.Module):
    """
    더 깊은 인코더/디코더 스택을 가진 변형 U-Net.
    - enc를 5단계까지 내려가고, bottleneck 채널 수를 늘려 더 풍부한 표현을 학습
    - 기존 블록(ResidualConv, Down, Up, ASPP)을 재사용하므로 호환성 유지
    사용법은 ImprovedUNet과 동일하되 클래스 이름만 DeepUNet으로 변경하면 됨.
    """
    def __init__(self, in_channel, out_channel, img_size: Optional[tuple] = None,
                 base_ch=32, norm="bn", se=True, drop=0.1, use_aspp=True):
        super().__init__()
        self.img_size = img_size

        # Encoder (5단계)
        self.enc1 = ResidualConv(in_channel,      base_ch,       norm=norm, se=se, drop=0.0)
        self.enc2 = Down(base_ch,                 base_ch * 2,   norm=norm, se=se, drop=drop)
        self.enc3 = Down(base_ch * 2,             base_ch * 4,   norm=norm, se=se, drop=drop)
        self.enc4 = Down(base_ch * 4,             base_ch * 8,   norm=norm, se=se, drop=drop)
        self.enc5 = Down(base_ch * 8,             base_ch * 16,  norm=norm, se=se, drop=drop)

        # Bottleneck
        self.bottleneck = ResidualConv(base_ch * 16, base_ch * 32, norm=norm, se=se, drop=drop)
        self.aspp = ASPP(base_ch * 32, base_ch * 32, rates=(1, 6, 12, 18), norm=norm) if use_aspp else nn.Identity()

        # Decoder (5단계 up)
        self.up4 = Up(base_ch * 32, base_ch * 16, norm=norm, se=se, drop=drop)   # skip: enc5
        self.up3 = Up(base_ch * 16, base_ch * 8,  norm=norm, se=se, drop=drop)   # skip: enc4
        self.up2 = Up(base_ch * 8,  base_ch * 4,  norm=norm, se=se, drop=drop)   # skip: enc3
        self.up1 = Up(base_ch * 4,  base_ch * 2,  norm=norm, se=se, drop=drop)   # skip: enc2
        self.up0 = Up(base_ch * 2,  base_ch,      norm=norm, se=se, drop=0.0)    # skip: enc1

        self.head = nn.Conv2d(base_ch, out_channel, kernel_size=1)

        # Kaiming init
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # encoder
        e1 = self.enc1(x)       # H
        e2 = self.enc2(e1)      # H/2
        e3 = self.enc3(e2)      # H/4
        e4 = self.enc4(e3)      # H/8
        e5 = self.enc5(e4)      # H/16

        # bottleneck + ASPP
        b = self.bottleneck(e5)
        b = self.aspp(b)

        # decoder with attention gates & skips
        d4 = self.up4(b,  e5)   # H/16
        d3 = self.up3(d4, e4)   # H/8
        d2 = self.up2(d3, e3)   # H/4
        d1 = self.up1(d2, e2)   # H/2
        d0 = self.up0(d1, e1)   # H

        logits = self.head(d0)
        return logits
