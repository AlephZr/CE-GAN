from torch import nn
import functools
import torch
from torch.utils.tensorboard import SummaryWriter


class DCGANGenerator(nn.Module):
    def __init__(self, nz, ngf, nc):
        super(DCGANGenerator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 4 x 4
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 8 x 8
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf*1) x 16 x 16
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 32 x 32
        )

    def forward(self, input):
        output = self.main(input)
        return output


class DCGANDiscriminator(nn.Module):
    def __init__(self, ndf, nc):
        super(DCGANDiscriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 32 x 32
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 16 x 16
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 8 x 8
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 4 x 4
            nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),
            # nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input)

        return output.view(-1, 1).squeeze(1)


# class DCGANGenerator(nn.Module):
#     def __init__(self, nz, ngf, nc):
#         super(DCGANGenerator, self).__init__()
#         self.main = nn.Sequential(
#             # input is Z, going into a convolution
#             nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
#             nn.BatchNorm2d(ngf * 8),
#             nn.ReLU(True),
#             # state size. (ngf*8) x 4 x 4
#             nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ngf * 4),
#             nn.ReLU(True),
#             # state size. (ngf*4) x 8 x 8
#             nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ngf * 2),
#             nn.ReLU(True),
#             # state size. (ngf*2) x 16 x 16
#             nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ngf),
#             nn.ReLU(True),
#             # state size. (ngf) x 32 x 32
#             nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
#             nn.Tanh()
#             # state size. (nc) x 64 x 64
#         )
#
#     def forward(self, input):
#         output = self.main(input)
#         return output
#
#
# class DCGANDiscriminator(nn.Module):
#     def __init__(self, ndf, nc):
#         super(DCGANDiscriminator, self).__init__()
#         self.main = nn.Sequential(
#             # input is (nc) x 64 x 64
#             nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf) x 32 x 32
#             nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ndf * 2),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf*2) x 16 x 16
#             nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ndf * 4),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf*4) x 8 x 8
#             nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ndf * 8),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf*8) x 4 x 4
#             nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
#             # nn.Sigmoid()
#         )
#
#     def forward(self, input):
#         output = self.main(input)
#
#         return output.view(-1, 1).squeeze(1)


class EGANGenerator_32(nn.Module):
    def __init__(self, z_dim, ngf=64, output_nc=3, norm_layer=nn.BatchNorm2d):
        super(EGANGenerator_32, self).__init__()

        self.z_dim = z_dim
        self.ngf = ngf
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        use_bias = True
        seq = [nn.ConvTranspose2d(z_dim, ngf * 4, 4, stride=1, padding=0, bias=use_bias),
               norm_layer(ngf * 4),
               nn.ReLU(),
               nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, stride=2, padding=(1, 1), bias=use_bias),
               norm_layer(ngf * 2),
               nn.ReLU(),
               nn.ConvTranspose2d(ngf * 2, ngf, 4, stride=2, padding=(1, 1), bias=use_bias),
               norm_layer(ngf),
               nn.ReLU(),
               nn.ConvTranspose2d(ngf, output_nc, 4, stride=2, padding=(1, 1)),
               nn.Tanh()]

        self.model = nn.Sequential(*seq)

    def forward(self, model_input):
        return self.model(model_input.view(-1, self.z_dim, 1, 1))


class EGANDiscriminator_32(nn.Module):
    def __init__(self, ndf=64, input_nc=3, norm_layer=nn.BatchNorm2d):
        super(EGANDiscriminator_32, self).__init__()

        self.ndf = ndf
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        use_bias = True
        seq = [nn.Conv2d(input_nc, ndf, 4, stride=2, padding=(1, 1), bias=use_bias),
               # norm_layer(ndf),
               nn.LeakyReLU(0.2),
               nn.Conv2d(ndf, ndf * 2, 4, stride=2, padding=(1, 1), bias=use_bias),
               norm_layer(ndf * 2),
               nn.LeakyReLU(0.2),
               nn.Conv2d(ndf * 2, ndf * 4, 4, stride=2, padding=(1, 1), bias=use_bias),
               norm_layer(ndf * 4),
               nn.LeakyReLU(0.2)]

        self.cnn_model = nn.Sequential(*seq)

        fc = [nn.Linear(4 * 4 * ndf * 4, 1)]
        self.fc = nn.Sequential(*fc)

    def forward(self, model_input):
        x = self.cnn_model(model_input)
        x = x.view(-1, 4 * 4 * self.ndf * 4)
        x = self.fc(x)
        return x.squeeze(1)


class EGANGenerator_64(nn.Module):
    def __init__(self, z_dim, ngf=64, output_nc=3, norm_layer=nn.BatchNorm2d):
        super(EGANGenerator_64, self).__init__()

        self.z_dim = z_dim
        self.ngf = ngf
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        use_bias = True
        seq = [nn.ConvTranspose2d(z_dim, ngf * 8, 4, stride=2, padding=(1, 1), bias=use_bias),
               norm_layer(ngf * 8),
               nn.ReLU(),
               nn.ConvTranspose2d(ngf * 8, ngf * 8, 4, stride=2, padding=(1, 1), bias=use_bias),
               norm_layer(ngf * 8),
               nn.ReLU(),
               nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, stride=2, padding=(1, 1), bias=use_bias),
               norm_layer(ngf * 4),
               nn.ReLU(),
               nn.ConvTranspose2d(ngf * 4, ngf * 4, 4, stride=2, padding=(1, 1), bias=use_bias),
               norm_layer(ngf * 4),
               nn.ReLU(),
               nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, stride=2, padding=(1, 1), bias=use_bias),
               norm_layer(ngf * 2),
               nn.ReLU(),
               nn.ConvTranspose2d(ngf * 2, output_nc, 3, stride=1, padding=0),
               nn.Tanh()]

        self.model = nn.Sequential(*seq)

    def forward(self, model_input):
        return self.model(model_input.view(-1, self.z_dim, 1, 1))


class EGANDiscriminator_64(nn.Module):
    def __init__(self, ndf=64, input_nc=3, norm_layer=nn.BatchNorm2d):
        super(EGANDiscriminator_64, self).__init__()

        self.ndf = ndf
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        use_bias = True
        seq = [nn.Conv2d(input_nc, ndf, 4, stride=2, padding=(1, 1), bias=use_bias),
               # norm_layer(ndf),
               nn.LeakyReLU(0.2),
               nn.Conv2d(ndf, ndf * 2, 4, stride=2, padding=(1, 1), bias=use_bias),
               norm_layer(ndf * 2),
               nn.LeakyReLU(0.2),
               nn.Conv2d(ndf * 2, ndf * 4, 4, stride=2, padding=(1, 1), bias=use_bias),
               norm_layer(ndf * 4),
               nn.LeakyReLU(0.2),
               nn.Conv2d(ndf * 4, ndf * 8, 4, stride=2, padding=(1, 1), bias=use_bias),
               norm_layer(ndf * 8),
               nn.LeakyReLU(0.2)]

        self.cnn_model = nn.Sequential(*seq)

        fc = [nn.Linear(4 * 4 * ndf * 8, 1)]
        self.fc = nn.Sequential(*fc)

    def forward(self, model_input):
        x = self.cnn_model(model_input)
        x = x.view(-1, 4 * 4 * self.ndf * 8)
        x = self.fc(x)
        return x.squeeze(1)


class EGANGenerator_128(nn.Module):
    def __init__(self, z_dim, ngf=64, output_nc=3, norm_layer=nn.BatchNorm2d):
        super(EGANGenerator_128, self).__init__()

        self.z_dim = z_dim
        self.ngf = ngf
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        use_bias = True
        seq = [nn.ConvTranspose2d(z_dim, ngf * 16, 4, stride=2, padding=(1, 1), bias=use_bias),
               norm_layer(ngf * 8),
               nn.LeakyReLU(0.2),
               nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, stride=2, padding=(1, 1), bias=use_bias),
               norm_layer(ngf * 8),
               nn.LeakyReLU(0.2),
               nn.ConvTranspose2d(ngf * 8, ngf * 8, 4, stride=2, padding=(1, 1), bias=use_bias),
               norm_layer(ngf * 8),
               nn.LeakyReLU(0.2),
               nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, stride=2, padding=(1, 1), bias=use_bias),
               norm_layer(ngf * 4),
               nn.LeakyReLU(0.2),
               nn.ConvTranspose2d(ngf * 4, ngf * 4, 4, stride=2, padding=(1, 1), bias=use_bias),
               norm_layer(ngf * 4),
               nn.LeakyReLU(0.2),
               nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, stride=2, padding=(1, 1), bias=use_bias),
               norm_layer(ngf * 2),
               nn.LeakyReLU(0.2),
               nn.ConvTranspose2d(ngf * 2, output_nc, 3, stride=1, padding=0),
               nn.Tanh()]

        self.model = nn.Sequential(*seq)

    def forward(self, model_input):
        return self.model(model_input.view(-1, self.z_dim, 1, 1))


class EGANDiscriminator_128(nn.Module):
    def __init__(self, ndf=64, input_nc=3, norm_layer=nn.BatchNorm2d):
        super(EGANDiscriminator_128, self).__init__()

        self.ndf = ndf
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        use_bias = True
        seq = [nn.Conv2d(input_nc, ndf, 4, stride=2, padding=(1, 1), bias=use_bias),
               # norm_layer(ndf),
               nn.LeakyReLU(0.2),
               nn.Conv2d(ndf, ndf * 2, 4, stride=2, padding=(1, 1), bias=use_bias),
               norm_layer(ndf * 2),
               nn.LeakyReLU(0.2),
               nn.Conv2d(ndf * 2, ndf * 4, 4, stride=2, padding=(1, 1), bias=use_bias),
               norm_layer(ndf * 4),
               nn.LeakyReLU(0.2),
               nn.Conv2d(ndf * 4, ndf * 8, 4, stride=2, padding=(1, 1), bias=use_bias),
               norm_layer(ndf * 8),
               nn.LeakyReLU(0.2),
               nn.Conv2d(ndf * 8, ndf * 16, 4, stride=2, padding=(1, 1), bias=use_bias),
               norm_layer(ndf * 16),
               nn.LeakyReLU(0.2)]

        self.cnn_model = nn.Sequential(*seq)

        fc = [nn.Linear(4 * 4 * ndf * 16, 1)]
        self.fc = nn.Sequential(*fc)

    def forward(self, model_input):
        x = self.cnn_model(model_input)
        x = x.view(-1, 4 * 4 * self.ndf * 16)
        x = self.fc(x)
        return x.squeeze(1)


class WGANGenerator_cifar10(nn.Module):
    def __init__(self, input_dim, image_channels, hidden_dim):
        super(WGANGenerator_cifar10, self).__init__()
        self.input_dim = input_dim
        self.image_channels = image_channels
        self.hidden_dim = hidden_dim

        preprocess = nn.Sequential(
            nn.Linear(self.input_dim, 4 * 4 * 4 * self.hidden_dim),
            nn.BatchNorm1d(4 * 4 * 4 * self.hidden_dim),
            nn.ReLU(True),
        )

        block1 = nn.Sequential(
            nn.ConvTranspose2d(4 * self.hidden_dim, 2 * self.hidden_dim, 2, stride=2),
            nn.BatchNorm2d(2 * self.hidden_dim),
            nn.ReLU(True),
        )
        block2 = nn.Sequential(
            nn.ConvTranspose2d(2 * self.hidden_dim, self.hidden_dim, 2, stride=2),
            nn.BatchNorm2d(self.hidden_dim),
            nn.ReLU(True),
        )
        deconv_out = nn.ConvTranspose2d(self.hidden_dim, self.image_channels, 2, stride=2)

        self.preprocess = preprocess
        self.block1 = block1
        self.block2 = block2
        self.deconv_out = deconv_out
        self.tanh = nn.Tanh()

    def forward(self, model_input):
        output = self.preprocess(model_input)
        output = output.view(-1, 4 * self.hidden_dim, 4, 4)
        output = self.block1(output)
        output = self.block2(output)
        output = self.deconv_out(output)
        output = self.tanh(output)
        return output.view(-1, self.image_channels, 32, 32)


class WGANDiscriminator_cifar10(nn.Module):
    def __init__(self, image_channels, hidden_dim):
        super(WGANDiscriminator_cifar10, self).__init__()
        self.image_channels = image_channels
        self.hidden_dim = hidden_dim

        main = nn.Sequential(
            nn.Conv2d(self.image_channels, self.hidden_dim, 3, 2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(self.hidden_dim, 2 * self.hidden_dim, 3, 2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(2 * self.hidden_dim, 4 * self.hidden_dim, 3, 2, padding=1),
            nn.LeakyReLU(),
        )

        self.main = main
        self.linear = nn.Linear(4 * 4 * 4 * self.hidden_dim, 1)

    def forward(self, model_input):
        output = self.main(model_input)
        output = output.view(-1, 4 * 4 * 4 * self.hidden_dim)
        output = self.linear(output)
        return output.squeeze(1)


class FC2Generator(nn.Module):
    def __init__(self, nz=2, ngf=512):
        super(FC2Generator, self).__init__()

        main = nn.Sequential(
            nn.Linear(nz, ngf),
            nn.ReLU(True),
            nn.Linear(ngf, ngf),
            nn.ReLU(True),
            nn.Linear(ngf, ngf),
            nn.ReLU(True),
            nn.Linear(ngf, 2),
        )
        self.main = main

    def forward(self, noise):
        output = self.main(noise)
        return output


class FC2Discriminator(nn.Module):
    def __init__(self, ndf=512):
        super(FC2Discriminator, self).__init__()

        main = nn.Sequential(
            nn.Linear(2, ndf),
            nn.ReLU(True),
            nn.Linear(ndf, ndf),
            # nn.BatchNorm1d(ndf),
            nn.ReLU(True),
            # nn.BatchNorm1d(ndf),
            nn.Linear(ndf, ndf),
            nn.ReLU(True),
            nn.Linear(ndf, 1),
        )
        self.main = main

    def forward(self, inputs):
        output = self.main(inputs)
        return output.view(-1)
