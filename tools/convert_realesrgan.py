#!/usr/bin/env python3
"""RealESRGAN_x4plus.pth → RealESRGAN_x4.mlmodel (CoreML, image in/out).

Rebuilds the optional upscaler weight for LiToStudio's first-run installer:
512×512 image input → 2048×2048 image output, matching Sources/LiToKit/Upscaler.swift
(it feeds 32BGRA pixel buffers and reads an image output; CoreML handles the
BGRA↔RGB conversion declared via ImageType).

RRDBNet is defined inline (the standard 23-block x4plus architecture) so we don't
need basicsr. Output values are scaled to [0,255] inside the graph because CoreML
image outputs expect that range.
"""
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import coremltools as ct


class ResidualDenseBlock(nn.Module):
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(nf, gc)
        self.rdb2 = ResidualDenseBlock(nf, gc)
        self.rdb3 = ResidualDenseBlock(nf, gc)

    def forward(self, x):
        return self.rdb3(self.rdb2(self.rdb1(x))) * 0.2 + x


class RRDBNet(nn.Module):
    def __init__(self, nf=64, nb=23, gc=32):
        super().__init__()
        self.conv_first = nn.Conv2d(3, nf, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(nf, gc) for _ in range(nb)])
        self.conv_body = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_hr = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_last = nn.Conv2d(nf, 3, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        feat = self.conv_first(x)
        feat = feat + self.conv_body(self.body(feat))
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))


class Wrapped(nn.Module):
    """[0,1] RGB in (via ImageType scale=1/255) → [0,255] RGB out (image output)."""
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return torch.clamp(self.net(x), 0, 1) * 255.0


def main():
    pth, out = sys.argv[1], sys.argv[2]
    state = torch.load(pth, map_location="cpu", weights_only=True)
    state = state.get("params_ema", state)
    net = RRDBNet()
    net.load_state_dict(state, strict=True)
    net.eval()
    model = Wrapped(net).eval()

    size = 512
    example = torch.rand(1, 3, size, size)
    with torch.no_grad():
        traced = torch.jit.trace(model, example)

    ml = ct.convert(
        traced,
        inputs=[ct.ImageType(name="input", shape=(1, 3, size, size),
                             scale=1 / 255.0, color_layout=ct.colorlayout.RGB)],
        outputs=[ct.ImageType(name="output", color_layout=ct.colorlayout.RGB)],
        convert_to="neuralnetwork",
        minimum_deployment_target=ct.target.macOS13,
    )
    ml.short_description = "Real-ESRGAN x4plus (512->2048), rebuilt from official weights"
    ml.save(out)
    print("saved", out)


if __name__ == "__main__":
    main()
