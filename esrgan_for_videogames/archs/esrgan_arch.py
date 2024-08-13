from typing import Tuple
import torch
from torch import nn as nn
from torch.nn import functional as F

from basicsr.archs.arch_util import default_init_weights
from basicsr.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class ExampleArch(nn.Module):
    """Example architecture.

    Args:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_out_ch (int): Channel number of outputs. Default: 3.
        num_feat (int): Channel number of intermediate features. Default: 64.
        upscale (int): Upsampling factor. Default: 4.
    """

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, upscale=4):
        super(ExampleArch, self).__init__()
        self.upscale = upscale

        self.conv1 = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
        self.upconv2 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(2)

        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # initialization
        default_init_weights(
            [self.conv1, self.conv2, self.conv3, self.upconv1, self.upconv2, self.conv_hr, self.conv_last], 0.1)

    def forward(self, x):
        feat = self.lrelu(self.conv1(x))
        feat = self.lrelu(self.conv2(feat))
        feat = self.lrelu(self.conv3(feat))

        out = self.lrelu(self.pixel_shuffle(self.upconv1(feat)))
        out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))

        out = self.conv_last(self.lrelu(self.conv_hr(out)))
        base = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)
        out += base
        return out


class DenseBlock(nn.Module):
    def __init__(self,
                 nf: int = 64,
                 gc: int = 32,
                 residual_scaling: float = 0.2,
                 kernel_size: int = 3,
                 padding: int = 1,
                 stride: int = 1,
                 n_res_blocks: int = 5) -> None:
        """
            Parameters
            ----------
            nf: int
                Количество входных и выходных каналов блока.
            gc: int
                Количество скрытых каналов блока.
            residual_scaling: float
                Коэффициент умножения выходов блока перед сложением с предыдущим слоем.

            Returns
            -------
            None
        """
        super().__init__()
        self.residual_scaling = residual_scaling
        self.n_res_blocks = n_res_blocks
        self.cat_results = []

        self.res_blocks = []
        for i in range(self.n_res_blocks):
            self.res_blocks.append(
                ResBlockWoBN(nf + i * gc , gc)
            )

        self.conv_last = nn.Conv2d(nf + (self.n_res_blocks - 1) * gc, nf, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """
            Parameters
            ----------
            x: torch.FloatTensor
                Входной тензор формата (bs, c, h, w).

            Returns
            -------
            torch.FloatTensor
                Выходной тензор, получается путем применения слоев к входному тензору.
        """
        self.cat_results = []
        out = self.res_blocks[0](x)
        self.cat_results.append(out)

        for i in range(1, len(self.res_blocks) - 1):
            cat = torch.cat((x, *self.cat_results[:i]), dim=1)
            out = self.res_blocks[i](cat)
            self.cat_results.append(out)
        out = self.conv_last(torch.cat((x, *self.cat_results), dim=1))
        out = self.residual_scaling * out + x

        return out


class ResBlockWoBN(nn.Module):
    def __init__(
        self,
        nf: int = 64,
        gc: int = 32,
        kernel_size: int = 3,
        padding: int = 1,
        stride: int = 1) -> None:
        """
            Residual блок без Batch Normalization: Conv -> LReLU -> F(x) + x.

            Returns
            -------
            None
        """
        super().__init__()
        self.nf = nf
        self.conv = nn.Conv2d(nf, gc, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.lrelu = nn.LeakyReLU()

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """
            Parameters
            ----------
            x: torch.FloatTensor
                Входной тензор формата (bs, c, h, w).

            Returns
            -------
            torch.FloatTensor
                Выходной тензор, получается путем применения слоев к входному тензору.
        """
        out = self.conv(x)
        out = self.lrelu(out)

        return out


class RRDB(nn.Module):
    def __init__(self,
                 nf: int = 64,
                 gc: int = 32,
                 residual_scaling: float = 0.2,
                 n_dense_blocks: int = 5,
                 n_res_blocks: int = 5) -> None:
        """
            Parameters
            ----------
            nf: int
                Количество входных и выходных каналов блока.
            gc: int
                Количество скрытых каналов блока.
            residual_scaling: float
                Коэффициент умножения выходов блока перед сложением с предыдущим слоем.

            Returns
            -------
            None
        """
        super().__init__()
        self.residual_scaling = residual_scaling
        self.rrd_blocks = []
        for _ in range(n_dense_blocks):
          self.rrd_blocks.append(DenseBlock(nf, gc, residual_scaling, n_res_blocks=n_res_blocks))


    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """
            Parameters
            ----------
            x: torch.FloatTensor
                Входной тензор формата (bs, c, h, w).

            Returns
            -------
            torch.FloatTensor
                Выходной тензор, получается путем применения слоев к входному тензору.
        """
        out = self.rrd_blocks[0](x)
        for rrd_block in self.rrd_blocks[1:]:
          out = rrd_block(out)
        out = self.residual_scaling * out + x

        return out


class ESRGAN_Up(nn.Module):
    def __init__(self,
                 factor: int,
                 mode: str = "nearest",
                 in_channels: int = 32,
                 kernel_size: int = 3,
                 padding: int = 1,
                 stride: int = 1) -> None:
        """
            Parameters
            ----------
            factor: int
                Коэффициент увеличения [2|3|4].
            mode: str
                Тип интерполяции [bicubic|bilinear|nearest].

            Returns
            -------
            None
        """
        super().__init__()
        self.factor = factor
        self.mode = mode

        if self.factor in [2, 3]:
          self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        if self.factor == 4:
          self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
          self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """
            Parameters
            ----------
            x: torch.FloatTensor
                Входной тензор формата (bs, c, h, w).

            Returns
            -------
            torch.FloatTensor
                Выходной тензор, получается путем применения слоев к входному тензору.
        """
        if self.factor in [2, 3]:
          out = self.conv(x)
          inter_out = torch.nn.functional.interpolate(
              out, scale_factor=self.factor, mode=self.mode
              )
          return inter_out
        elif self.factor == 4:
          out = self.conv1(x)
          inter_out = torch.nn.functional.interpolate(
              out, scale_factor=2, mode=self.mode
              )
          out = self.conv2(inter_out)
          inter_out = torch.nn.functional.interpolate(
              out, scale_factor=2, mode=self.mode
              )
          return inter_out


@ARCH_REGISTRY.register()
class ESRGAN_G(nn.Module):
    def __init__(
            self,
            nchannels: int = 3,
            nblocks: int = 16,
            n_res_blocks: int = 5,
            n_dense_blocks: int = 5,
            nf: int = 64,
            gc: int = 32,
            scale: int = 4,
            residual_scaling: float = 0.2,
            interpolation_mode: str = "nearest",
            kernel_size: int = 3,
            padding: int = 1,
            stride: int = 1
        ) -> None:
        """
            Parameters
            ----------
            nchannels: int
                Количество каналов входного изображения.
            nblocks: int
                Количество RRDB блоков.
            nf: int
                Количество входных и выходных каналов для RRDB блоков.
            gc: int
                Количество скрытых каналов для RRDB блоков.
            scale: int
                Коэффициент увеличения [2|3|4].
            residual_scaling: float
                Коэффициент умножения выходов блока перед сложением с предыдущим слоем.
            interpolation_mode: str
                Тип интерполяции для upsample блоков [bicubic|bilinear|nearest].

            Returns
            -------
            None
        """
        super().__init__()
        self.conv_first = nn.Conv2d(nchannels, nf, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.basic_blocks = []
        for i in range(nblocks):
          self.basic_blocks.append(RRDB(
              nf=nf, gc=gc, residual_scaling=residual_scaling, n_dense_blocks=n_dense_blocks, n_res_blocks=n_res_blocks
          ))
        self.conv_blocks = nn.Conv2d(nf, nf, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.upsampling = ESRGAN_Up(scale, mode=interpolation_mode, in_channels=nf)
        self.conv_upcorr = nn.Conv2d(nf, nchannels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.conv_last = nn.Conv2d(nchannels, nchannels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)


    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """
            Parameters
            ----------
            x: torch.FloatTensor
                Входной тензор формата (bs, c, h, w).

            Returns
            -------
            torch.FloatTensor
                Выходной тензор, получается путем применения слоев к входному тензору.
        """
        out_conv_first = self.conv_first(x)
        out_basic_blocks = self.basic_blocks[0](out_conv_first)
        for basic_block in self.basic_blocks[1:]:
          out_basic_blocks = basic_block(out_basic_blocks)
        out_conv_blocks = self.conv_blocks(out_basic_blocks)

        out = out_conv_first + out_conv_blocks
        out = self.upsampling(out)
        out = self.conv_upcorr(out)
        out = self.conv_last(out)

        return out


class DiscriminatorBlock(nn.Module):
    def __init__(self, inf: int,
                 outf: int,
                 stride: int,
                 kernel_size: int = 3,
                 padding: int = 1,
                 use_bn: bool = True) -> None:
        """
            Блок дискриминатора SRGAN: Conv -> BN -> LeakyReLU.

            Parameters
            ----------
            inf: int
                Количество каналов входного изображения.
            outf: int
                Количество каналов выходного изображения.
            stride: int
                Шаг свертки [1|2].
            use_bn: bool
                Флаг использования слоя BN.

            Returns
            -------
            None
        """
        super().__init__()
        self.use_bn = use_bn
        self.conv_first = nn.Conv2d(inf, outf, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(outf)
        self.lrelu = nn.LeakyReLU()

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """
            Parameters
            ----------
            x: torch.FloatTensor
                Входной тензор формата (bs, c, h, w).

            Returns
            -------
            torch.FloatTensor
                Выходной тензор, получается путем применения слоев к входному тензору.
        """
        out = self.conv_first(x)
        if self.use_bn:
            out = self.bn(out)

        out = self.lrelu(out)
        return out


@ARCH_REGISTRY.register()
class ESRGAN_D(nn.Module):
    def __init__(self,
                 nchannels: int = 3,
                 out_channels: int = 64,
                 nblocks: int = 8,
                 dense_neurons: int = 1024,
                 hq_size: int = 32,
                 use_sigmoid: bool = True) -> None:
        """
            Parameters
            ----------
            nchannels: int
                Количество каналов входного изображения.
            nblocks: int
                Количество DiscriminatorBlock блоков.
            use_sigmoid: bool
                Флаг использования сигмоидальной функции активации после последнего слоя.

            Returns
            -------
            None
        """
        super().__init__()
        self.use_sigmoid = use_sigmoid
        self.hq_size = hq_size
        self.basic_blocks = []
        self.basic_blocks.append(DiscriminatorBlock(nchannels, out_channels, stride=1, use_bn=False))
        self.basic_blocks.append(DiscriminatorBlock(out_channels, out_channels * 2, stride=2))
        self.hq_size //= 2
        for i in range(1, nblocks // 2):
            curr_out_channels = out_channels * (2 ** i)
            self.basic_blocks.append(
                DiscriminatorBlock(curr_out_channels, curr_out_channels, stride=1)
            )
            if self.hq_size > 2:
                self.basic_blocks.append(
                    DiscriminatorBlock(curr_out_channels, curr_out_channels * 2, stride=2)
                )
                self.hq_size //= 2
            else:
                break
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(curr_out_channels * 2 * (self.hq_size**2), dense_neurons)
        self.lrelu = nn.LeakyReLU()
        self.dense_out = nn.Linear(dense_neurons, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """
            Parameters
            ----------
            x: torch.FloatTensor
                Входной тензор формата (bs, c, h, w).

            Returns
            -------
            torch.FloatTensor
                Выходной тензор, получается путем применения слоев к входному тензору.
        """
        out = self.basic_blocks[0](x)
        out = self.basic_blocks[1](out)
        for basic_block in self.basic_blocks[2:]:
            out = basic_block(out)
        out = torch.flatten(out, 1)
        out = self.dense(out)
        out = self.lrelu(out)
        out = self.dense_out(out)
        if self.use_sigmoid:
            out = self.sigmoid(out)
        return out