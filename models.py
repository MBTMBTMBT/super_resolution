import torch
import torch.nn.functional
import enum
from torchstat import stat


@enum.unique
class ModelSelect(enum.IntEnum):
    SRCNN = 0
    SRCNN_LINEAR = 1
    ESRGAN = 2
    FSRCNN = 3


@enum.unique
class DiscriminatorSelect(enum.IntEnum):
    NO_DISCRIMINATOR = 0
    PATCH_DISCRIMINATOR_SINGLE_INPUT = 1
    PATCH_DISCRIMINATOR_DOUBLE_INPUT = 2


class SRCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.up_sample = torch.nn.UpsamplingBilinear2d(scale_factor=2)
        self.patch_extraction \
            = torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(5, 5), stride=(1, 1), padding=2)
        self.non_linear \
            = torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(1, 1), stride=(1, 1), padding=0)
        self.reconstruction \
            = torch.nn.Conv2d(in_channels=32, out_channels=3, kernel_size=(9, 9), stride=(1, 1), padding=4)

    def forward(self, x):
        x = self.up_sample(x)
        fm_1 = torch.nn.functional.relu(self.patch_extraction(x))
        fm_2 = torch.nn.functional.relu(self.non_linear(fm_1))
        fm_3 = torch.nn.functional.sigmoid(self.reconstruction(fm_2))
        return fm_3


# https://github.com/yjn870/SRCNN-pytorch
class SRCNNLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.up_sample = torch.nn.UpsamplingBilinear2d(scale_factor=2)
        self.patch_extraction \
            = torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(5, 5), stride=(1, 1), padding=2)
        self.non_linear \
            = torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(1, 1), stride=(1, 1), padding=0)
        self.reconstruction \
            = torch.nn.Conv2d(in_channels=32, out_channels=3, kernel_size=(9, 9), stride=(1, 1), padding=4)

    def forward(self, x):
        x = self.up_sample(x)
        fm_1 = torch.nn.functional.relu(self.patch_extraction(x))
        fm_2 = torch.nn.functional.relu(self.non_linear(fm_1))
        fm_3 = self.reconstruction(fm_2)
        return fm_3


# https://github.com/yjn870/FSRCNN-pytorch
class FSRCNN(torch.nn.Module):
    def __init__(self, scale_factor=2, num_channels=3, d=56, s=12, m=4):
        super(FSRCNN, self).__init__()
        self.first_part = torch.nn.Sequential(
            torch.nn.Conv2d(num_channels, d, kernel_size=(5, 5), padding=5//2),
            torch.nn.PReLU(d)
        )
        self.mid_part = [torch.nn.Conv2d(d, s, kernel_size=(1, 1)), torch.nn.PReLU(s)]
        for _ in range(m):
            self.mid_part.extend([torch.nn.Conv2d(s, s, kernel_size=(3, 3), padding=3//2), torch.nn.PReLU(s)])
        self.mid_part.extend([torch.nn.Conv2d(s, d, kernel_size=(1, 1)), torch.nn.PReLU(d)])
        self.mid_part = torch.nn.Sequential(*self.mid_part)
        self.last_part = torch.nn.ConvTranspose2d(d, num_channels, kernel_size=(9, 9), stride=scale_factor,
                                                  padding=(9//2, 9//2), output_padding=scale_factor-1)

    def forward(self, x):
        x = self.first_part(x)
        x = self.mid_part(x)
        x = self.last_part(x)
        return x


def discriminator_block(in_filters, out_filters, normalization=True):
    """Returns downsampling layers of each discriminator block"""
    layers = [torch.nn.Conv2d(in_filters, out_filters, (4, 4), stride=(2, 2), padding=1)]
    if normalization:
        layers.append(torch.nn.InstanceNorm2d(out_filters))
    layers.append(torch.nn.LeakyReLU(0.2))
    return layers


class PatchDiscriminatorSingleInput(torch.nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(PatchDiscriminatorSingleInput, self).__init__()

        self.model = torch.nn.Sequential(
            *discriminator_block(in_channels, 32, normalization=False),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            torch.nn.ZeroPad2d((1, 0, 1, 0)),
            torch.nn.Conv2d(512, out_channels, (4, 4), padding=1, bias=False)
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return self.model(img)


class PatchDiscriminatorDoubleInput(torch.nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(PatchDiscriminatorDoubleInput, self).__init__()

        self.big_input = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 32, (3, 3), stride=(2, 2), padding=1),
            torch.nn.LeakyReLU(0.2),
        )
        self.small_input = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 32, (3, 3), stride=(1, 1), padding=1),
            torch.nn.LeakyReLU(0.2),
        )
        self.model = torch.nn.Sequential(
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            *discriminator_block(512, 1024),
            torch.nn.ZeroPad2d((1, 0, 1, 0)),
            torch.nn.Conv2d(1024, out_channels, (4, 4), padding=1, bias=False)
        )

    def forward(self, small_img: torch.Tensor, big_img: torch.Tensor) -> torch.Tensor:
        return self.model(torch.cat((self.big_input(big_img), self.small_input(small_img)), dim=1))


# https://github.com/wonbeomjang/ESRGAN-pytorch
class ResidualDenseBlock(torch.nn.Module):
    def __init__(self, nf, gc=32, res_scale=0.2):
        super(ResidualDenseBlock, self).__init__()
        self.layer1 = torch.nn.Sequential(torch.nn.Conv2d(nf + 0 * gc, gc, (3, 3), padding=1, bias=True), torch.nn.LeakyReLU())
        self.layer2 = torch.nn.Sequential(torch.nn.Conv2d(nf + 1 * gc, gc, (3, 3), padding=1, bias=True), torch.nn.LeakyReLU())
        self.layer3 = torch.nn.Sequential(torch.nn.Conv2d(nf + 2 * gc, gc, (3, 3), padding=1, bias=True), torch.nn.LeakyReLU())
        self.layer4 = torch.nn.Sequential(torch.nn.Conv2d(nf + 3 * gc, gc, (3, 3), padding=1, bias=True), torch.nn.LeakyReLU())
        self.layer5 = torch.nn.Sequential(torch.nn.Conv2d(nf + 4 * gc, nf, (3, 3), padding=1, bias=True), torch.nn.LeakyReLU())
        self.res_scale = res_scale

    def forward(self, x):
        layer1 = self.layer1(x)
        layer2 = self.layer2(torch.cat((x, layer1), 1))
        layer3 = self.layer3(torch.cat((x, layer1, layer2), 1))
        layer4 = self.layer4(torch.cat((x, layer1, layer2, layer3), 1))
        layer5 = self.layer5(torch.cat((x, layer1, layer2, layer3, layer4), 1))
        return layer5.mul(self.res_scale) + x


class ResidualInResidualDenseBlock(torch.nn.Module):
    def __init__(self, nf, gc=32, res_scale=0.2):
        super(ResidualInResidualDenseBlock, self).__init__()
        self.layer1 = ResidualDenseBlock(nf, gc)
        self.layer2 = ResidualDenseBlock(nf, gc)
        self.layer3 = ResidualDenseBlock(nf, gc, )
        self.res_scale = res_scale

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        return out.mul(self.res_scale) + x


def upsample_block(nf, scale_factor=2):
    block = []
    for _ in range(scale_factor//2):
        block += [
            torch.nn.Conv2d(nf, nf * (2 ** 2), (1, 1)),
            torch.nn.PixelShuffle(2),
            torch.nn.ReLU()
        ]
    return torch.nn.Sequential(*block)


class ESRGAN(torch.nn.Module):
    def __init__(self, in_channels=3, out_channels=3, nf=64, gc=32, scale_factor=2, n_basic_block=4):
        super(ESRGAN, self).__init__()
        self.conv1 = torch.nn.Sequential(torch.nn.ReflectionPad2d(1), torch.nn.Conv2d(in_channels, nf, (3, 3)), torch.nn.ReLU())
        basic_block_layer = []
        for _ in range(n_basic_block):
            basic_block_layer += [ResidualInResidualDenseBlock(nf, gc)]

        self.basic_block = torch.nn.Sequential(*basic_block_layer)
        self.conv2 = torch.nn.Sequential(torch.nn.ReflectionPad2d(1), torch.nn.Conv2d(nf, nf, (3, 3)), torch.nn.ReLU())
        self.upsample = upsample_block(nf, scale_factor=scale_factor)
        self.conv3 = torch.nn.Sequential(torch.nn.ReflectionPad2d(1), torch.nn.Conv2d(nf, nf, (3, 3)), torch.nn.ReLU())
        self.conv4 = torch.nn.Sequential(torch.nn.ReflectionPad2d(1), torch.nn.Conv2d(nf, out_channels, (3, 3)), torch.nn.ReLU())

    def forward(self, x):
        x1 = self.conv1(x)
        x = self.basic_block(x1)
        x = self.conv2(x)
        x = self.upsample(x + x1)
        x = self.conv3(x)
        x = self.conv4(x)
        return x


class ESRGANDiscriminator(torch.nn.Module):
    def __init__(self, num_conv_block=4):
        super(ESRGANDiscriminator, self).__init__()
        block = []
        in_channels = 3
        out_channels = 64
        for _ in range(num_conv_block):
            block += [torch.nn.ReflectionPad2d(1),
                      torch.nn.Conv2d(in_channels, out_channels, (3, 3)),
                      torch.nn.LeakyReLU(),
                      torch.nn.BatchNorm2d(out_channels)]
            in_channels = out_channels
            block += [torch.nn.ReflectionPad2d(1),
                      torch.nn.Conv2d(in_channels, out_channels, (3, 3), (2, 2)),
                      torch.nn.LeakyReLU()]
            out_channels *= 2
        out_channels //= 2
        in_channels = out_channels
        block += [torch.nn.Conv2d(in_channels, out_channels, (3, 3)),
                  torch.nn.LeakyReLU(0.2),
                  torch.nn.Conv2d(out_channels, out_channels, (3, 3))]
        self.feature_extraction = torch.nn.Sequential(*block)
        self.avgpool = torch.nn.AdaptiveAvgPool2d((512, 512))
        self.classification = torch.nn.Sequential(
            torch.nn.Linear(8192, 100),
            torch.nn.Linear(100, 1)
        )

    def forward(self, x):
        x = self.feature_extraction(x)
        x = x.view(x.size(0), -1)
        x = self.classification(x)
        return x


def weight_init(m: torch.nn.Module):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, torch.nn.BatchNorm2d):
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)


if __name__ == '__main__':
    test_model = FSRCNN(2)
    img_1 = torch.rand((1, 3, 240, 240))
    # img_2 = torch.rand((8, 3, 480, 480))
    # img = torch.rand((8, 3, 270, 480))
    # out = test_down(img)
    # print(out.shape)
    # out = test_up(out)
    # print(out.shape)
    # test_full.apply(weight_init)
    # test_full.train()
    out = test_model(img_1)
    print(out.shape)
    # stat(test_full, (3, 135, 240))
