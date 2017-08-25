#coding=utf8
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

###############################################################################
# Functions
###############################################################################


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1 or  classname.find('InstanceNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def get_norm_layer(norm_type):
    if norm_type == 'batch':
        norm_layer = nn.BatchNorm2d
    elif norm_type == 'instance':
        norm_layer = nn.InstanceNorm2d
    else:
        print('normalization layer [%s] is not found' % norm)
    return norm_layer

def define_G(input_nc, output_nc, ngf, which_model_netG, norm='batch', use_dropout=False, gpu_ids=[]):
    netG = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert(torch.cuda.is_available())

    if which_model_netG == 'resnet_9blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9, gpu_ids=gpu_ids)
    elif which_model_netG == 'resnet_6blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6, gpu_ids=gpu_ids)
    elif which_model_netG == 'unet_128':
        netG = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)
    elif which_model_netG == 'unet_256':
        #netG = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)
        netG = siamese_Unet_3(input_nc, output_nc, use_bn=True)
    elif which_model_netG == 'snet_256':
        netG = SpatialGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)
    elif which_model_netG == 'flownet':
        netG = FlowNetGenerator(input_nc, output_nc, gpu_ids=gpu_ids)
    elif which_model_netG == 'sia_unet':
        netG = siamese_Unet_4(input_nc, output_nc, use_bn=True)
    else:
        print('Generator model name [%s] is not recognized' % which_model_netG)
    if len(gpu_ids) > 0:
        netG.cuda(device_id=gpu_ids[0])
    netG.apply(weights_init)
    return netG


def define_D(input_nc, ndf, which_model_netD,
             n_layers_D=3, norm='batch', use_sigmoid=False, gpu_ids=[]):
    netD = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert(torch.cuda.is_available())
    if which_model_netD == 'basic':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
        #netD = NLayerDiscriminator2(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    elif which_model_netD == 'n_layers':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
        #netD = NLayerDiscriminator2(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    else:
        print('Discriminator model name [%s] is not recognized' %
              which_model_netD)
    if use_gpu:
        netD.cuda(device_id=gpu_ids[0])
    netD.apply(weights_init)
    return netD


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


##############################################################################
# Classes
##############################################################################


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)

class POSELoss(nn.Module):
    def __init__(self):
        super(POSELoss, self).__init__()
        self.loss = nn.L1Loss()

    def mask_pose(self, skeleton, fake, real_B):
        return (skeleton + 1) * fake, (skeleton + 1) * real_B

    def __call__(self, skeleton, fake, real_B):
        fake_, real_B_ = self.mask_pose(skeleton, fake, real_B)
        return self.loss(fake_, real_B_)

class TVRegularizerLoss(nn.Module):
    """ Enforces smoothness in image output. """

    def __init__(self, weight=1.0):
        # self.img_width = img_width
        # self.img_height = img_height
        self.weight = weight
        self.uses_learning_phase = False
        super(TVRegularizerLoss, self).__init__()

    def __call__(self, x):
        x_out = x.data
        img_width = x_out.size()[2]
        img_height = x_out.size()[3]
        a = np.square(x_out[:, :, :img_width - 1, :img_height - 1] - x_out[:, :, 1:, :img_height - 1])
        b = np.square(x_out[:, :, :img_width - 1, :img_height - 1] - x_out[:, :, :img_width - 1, 1:])
        # a = np.square(x_out[:, :self.img_width - 1, :self.img_height - 1, :] - x_out[:, 1:, :self.img_height - 1, :])
        # b = np.square(x_out[:, :self.img_width - 1, :self.img_height - 1, :] - x_out[:, :self.img_width - 1, 1:, :])
        #loss = self.weight * torch.mean(torch.sum(torch.pow(a + b, 1.25))) # 这个才正确
        loss = self.weight * torch.sum(torch.pow(a + b, 1.25))
        print loss
        return torch.FloatTensor(loss)

# class TRILoss(nn.Module):
#     def __init__(self, target_real_label=1.0, target_fake_label=0.0,
#                  tensor=torch.FloatTensor):
#         super(TRILoss, self).__init__()
#         self.real_label = target_real_label
#         self.fake_label = target_fake_label
#         self.real_label_var = None
#         self.fake_label_var = None
#         self.Tensor = tensor
#         if use_lsgan:
#             self.loss = nn.MSELoss()
#         else:
#             self.loss = nn.BCELoss()
#
#     def get_target_tensor(self, input, target_is_real):
#         target_tensor = None
#         if target_is_real:
#             create_label = ((self.real_label_var is None) or
#                             (self.real_label_var.numel() != input.numel()))
#             if create_label:
#                 real_tensor = self.Tensor(input.size()).fill_(self.real_label)
#                 self.real_label_var = Variable(real_tensor, requires_grad=False)
#             target_tensor = self.real_label_var
#         else:
#             create_label = ((self.fake_label_var is None) or
#                             (self.fake_label_var.numel() != input.numel()))
#             if create_label:
#                 fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
#                 self.fake_label_var = Variable(fake_tensor, requires_grad=False)
#             target_tensor = self.fake_label_var
#         return target_tensor
#
#     def __call__(self, input, target_is_real):
#         target_tensor = self.get_target_tensor(input, target_is_real)
#         return self.loss(input, target_tensor)


# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/
class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, gpu_ids=[]):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids

        model = [nn.Conv2d(input_nc, ngf, kernel_size=7, padding=3),
                 norm_layer(ngf, affine=True),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2, affine=True),
                      nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, 'zero', norm_layer=norm_layer, use_dropout=use_dropout)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                      norm_layer(int(ngf * mult / 2), affine=True),
                      nn.ReLU(True)]

        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=3)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout):
        conv_block = []
        p = 0
        # TODO: support padding types
        assert(padding_type == 'zero')
        p = 1

        # TODO: InstanceNorm
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim, affine=True),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim, affine=True)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(UnetGenerator, self).__init__()
        self.gpu_ids = gpu_ids

        # currently support only input_nc == output_nc
        assert(input_nc == output_nc)

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        if  self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost

        downconv = nn.Conv2d(outer_nc, inner_nc, kernel_size=4, stride=2, padding=1)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc, affine=True)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc, affine=True)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([self.model(x), x], 1)

# SpatialConvolution network
class SpatialGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(SpatialGenerator, self).__init__()
        self.gpu_ids = gpu_ids

        # currently support only input_nc == output_nc
        assert (input_nc == output_nc)

        # construct Spatial net structure
        snet_block = SpatialNetBlock(ngf * 8, ngf * 8, innermost=True)
        for i in range(num_downs - 5):
            snet_block = SpatialNetBlock(ngf * 8, ngf * 8, snet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        snet_block = SpatialNetBlock(ngf * 4, ngf * 8, snet_block, norm_layer=norm_layer)
        snet_block = SpatialNetBlock(ngf * 2, ngf * 4, snet_block, norm_layer=norm_layer)
        snet_block = SpatialNetBlock(ngf, ngf * 2, snet_block, norm_layer=norm_layer)
        snet_block = SpatialNetBlock(output_nc, ngf, snet_block, outermost=True, norm_layer=norm_layer)

        self.model = snet_block

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)

# spatial net block
class SpatialNetBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(SpatialNetBlock, self).__init__()
        self.outermost = outermost

        #nn.SpatialConvolution(self, nInputPlane, nOutputPlane, kW, kH, dW=1, dH=1, padW=0, padH=None):
        #downconv = nn.SpatialConvolution(inner_nc, outer_nc, 4, 4, 2, 2, 1, 1)
        downconv = nn.Conv2d(outer_nc, inner_nc, kernel_size=4, stride=2, padding=1)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc, affine=True)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc, affine=True)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downrelu, downconv]
            #down = [downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downrelu, downconv, downnorm]
            #down = [downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([self.model(x), x], 1)


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, gpu_ids=[]):
        super(NLayerDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids

        kw = 4
        padw = int(np.ceil((kw-1)/2))
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                                kernel_size=kw, stride=2, padding=padw),
                # TODO: use InstanceNorm
                norm_layer(ndf * nf_mult, affine=True),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                            kernel_size=kw, stride=1, padding=padw),
            # TODO: useInstanceNorm
            norm_layer(ndf * nf_mult, affine=True),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        if len(self.gpu_ids)  and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)

# Defines the PatchGAN discriminator2 with the specified arguments.
class NLayerDiscriminator2(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, gpu_ids=[]):
        super(NLayerDiscriminator2, self).__init__()
        self.gpu_ids = gpu_ids

        kw = 4
        padw = int(np.ceil((kw - 1) / 2))
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw),
                # TODO: use InstanceNorm
                norm_layer(ndf * nf_mult, affine=True),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw),
            # TODO: useInstanceNorm
            norm_layer(ndf * nf_mult, affine=True),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        # if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
        #     return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        # else:
        #     return self.model(input)
        return self.model(input)


# flownet
def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1,inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
            nn.LeakyReLU(0.1,inplace=True)
        )

def predict_flow(in_planes):
    return nn.Conv2d(in_planes, 2,kernel_size=3, stride=1, padding=1, bias=False)

def deconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=True),
        nn.LeakyReLU(0.1,inplace=True)
    )

def predict_result(in_planes):
    return nn.Conv2d(in_planes, 3,kernel_size=3, stride=1, padding=1, bias=False)

# FlowNet network
class FlowNetGenerator(nn.Module):
    #def __init__(self, input_nc, output_nc, ngf=64, ):
    #    super(FlowNetGenerator, self).__init__()
    #    # currently support only input_nc == output_nc
    #    assert (input_nc == output_nc)
    #def __init__(self, batchNorm=True):
    #    super(FlowNetGenerator, self).__init__()

    def __init__(self, inner_nc, outer_nc, batchNorm=True, gpu_ids=[]):
        super(FlowNetGenerator, self).__init__()
        self.gpu_ids = gpu_ids

        self.batchNorm = batchNorm

        #downconv = nn.Conv2d(outer_nc, inner_nc, kernel_size=4, stride=2, padding=1)

        self.conv0 = conv(self.batchNorm, inner_nc, 6, kernel_size=7, stride=2)

        #self.conv1 = conv(self.batchNorm, 3, 64, kernel_size=7, stride=2)
        self.conv1 = conv(self.batchNorm, 6, 64, kernel_size=7, stride=2)
        self.conv2 = conv(self.batchNorm, 64, 128, kernel_size=5, stride=2)
        self.conv3 = conv(self.batchNorm, 128, 256, kernel_size=5, stride=2)
        self.conv3_1 = conv(self.batchNorm, 256, 256)
        self.conv4 = conv(self.batchNorm, 256, 512, stride=2)
        self.conv4_1 = conv(self.batchNorm, 512, 512)
        self.conv5 = conv(self.batchNorm, 512, 512, stride=2)
        self.conv5_1 = conv(self.batchNorm, 512, 512)
        self.conv6 = conv(self.batchNorm, 512, 1024, stride=2)
        self.conv6_1 = conv(self.batchNorm, 1024, 1024)

        self.deconv5 = deconv(1024, 512)
        self.deconv4 = deconv(1026, 256)
        self.deconv3 = deconv(770, 128)
        self.deconv2 = deconv(386, 64)

        self.deconv1 = deconv(194, 32)
        self.deconv0 = deconv(98, 16)
        self.deconv0_0 = deconv(24, 8)

        self.predict_flow6 = predict_flow(1024)
        self.predict_flow5 = predict_flow(1026)
        self.predict_flow4 = predict_flow(770)
        self.predict_flow3 = predict_flow(386)
        self.predict_flow2 = predict_flow(194)

        self.predict_flow1 = predict_flow(98)
        self.predict_flow0 = predict_flow(24)
        #self.predict_flow0_0 = predict_flow(10)
        self.predict_result = predict_result(10)

        self.upsampled_flow6_to_5 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow5_to_4 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)

        self.upsampled_flow2_to_1 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow1_to_0 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow1_to_0_0 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)

    def forward(self, input):
        out_conv0 = self.conv0(input)
        out_conv1 = self.conv1(out_conv0)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))

        flow6 = self.predict_flow6(out_conv6)
        flow6_up = self.upsampled_flow6_to_5(flow6)
        out_deconv5 = self.deconv5(out_conv6)

        concat5 = torch.cat((out_conv5, out_deconv5, flow6_up), 1)
        flow5 = self.predict_flow5(concat5)
        flow5_up = self.upsampled_flow5_to_4(flow5)
        out_deconv4 = self.deconv4(concat5)

        concat4 = torch.cat((out_conv4, out_deconv4, flow5_up), 1)
        flow4 = self.predict_flow4(concat4)
        flow4_up = self.upsampled_flow4_to_3(flow4)
        out_deconv3 = self.deconv3(concat4)

        concat3 = torch.cat((out_conv3, out_deconv3, flow4_up), 1)
        flow3 = self.predict_flow3(concat3)
        flow3_up = self.upsampled_flow3_to_2(flow3)
        out_deconv2 = self.deconv2(concat3)

        concat2 = torch.cat((out_conv2, out_deconv2, flow3_up), 1)
        flow2 = self.predict_flow2(concat2)
        #------
        flow2_up = self.upsampled_flow2_to_1(flow2)
        out_deconv1 = self.deconv1(concat2)

        concat1 = torch.cat((out_conv1, out_deconv1, flow2_up), 1)
        flow1 = self.predict_flow1(concat1)
        flow1_up = self.upsampled_flow1_to_0(flow1)
        out_deconv0 = self.deconv0(concat1)

        concat0 = torch.cat((out_conv0, out_deconv0, flow1_up), 1)
        flow0 = self.predict_flow0(concat0)
        flow0_up = self.upsampled_flow1_to_0_0(flow0)
        out_deconv0_0 = self.deconv0_0(concat0)

        concat0_0 = torch.cat((out_deconv0_0, flow0_up), 1) #this out_deconv0_0
        #flow0 = self.predict_flow0_0(concat0_0)
        flow0 = self.predict_result(concat0_0)

        return flow0




def down_sample(dim_in, dim_out, kernel_size=4, stride=2, padding=1, bias=True, use_bn=False):
    if use_bn:
        return nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            #nn.BatchNorm2d(dim_out)
            nn.InstanceNorm2d(dim_out, affine=True)
        )
    else:
        return nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        )

def up_sample(dim_in, dim_out, kernel_size=4, stride=2, padding=1, bias=True, dropout=False):
    if dropout:
        return nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.InstanceNorm2d(dim_out, affine=True),
            nn.Dropout(0.5)
        )
    else:
        return nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.InstanceNorm2d(dim_out, affine=True)
        )

def up_sample_start(dim_in, dim_out, kernel_size=4, stride=2, padding=1, bias=True):
    return nn.Sequential(
        nn.ReLU(True),
        nn.ConvTranspose2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
        nn.BatchNorm2d(dim_out, affine=True)
    )

def up_sample_result(dim_in, dim_out, kernel_size=4, stride=2, padding=1, bias=True):
    return nn.Sequential(
        nn.ReLU(True),
        nn.ConvTranspose2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
        nn.Tanh()
    )


class siamese_Unet(nn.Module):
    def __init__(self, input_nc, output_nc, use_bn=False):
        super(siamese_Unet, self).__init__()
        self.gf_dim = 64

        ''' siamese down start'''
        # input 256*256*3
        self.down1_skeleton = down_sample(input_nc, self.gf_dim, use_bn=False)
        self.down2_skeleton = down_sample(self.gf_dim, self.gf_dim*2, use_bn=use_bn)
        self.down3_skeleton = down_sample(self.gf_dim*2, self.gf_dim*4, use_bn=use_bn)
        self.down4_skeleton = down_sample(self.gf_dim*4, self.gf_dim*8, use_bn=use_bn)
        self.down5_skeleton = down_sample(self.gf_dim*8, self.gf_dim*8, use_bn=use_bn)
        self.down6_skeleton = down_sample(self.gf_dim*8, self.gf_dim*8, use_bn=use_bn)
        self.down7_skeleton = down_sample(self.gf_dim*8, self.gf_dim*8, use_bn=use_bn)
        self.down8_skeleton = down_sample(self.gf_dim*8, self.gf_dim*8, use_bn=use_bn)
        # input 256*256*3
        self.down1_real = down_sample(input_nc, 64, use_bn=False)
        self.down2_real = down_sample(self.gf_dim, self.gf_dim*2, use_bn=use_bn)
        self.down3_real = down_sample(self.gf_dim*2, self.gf_dim*4, use_bn=use_bn)
        self.down4_real = down_sample(self.gf_dim*4, self.gf_dim*8, use_bn=use_bn)
        self.down5_real = down_sample(self.gf_dim*8, self.gf_dim*8, use_bn=use_bn)
        self.down6_real = down_sample(self.gf_dim*8, self.gf_dim*8, use_bn=use_bn)
        self.down7_real = down_sample(self.gf_dim*8, self.gf_dim*8, use_bn=use_bn)
        self.down8_real = down_sample(self.gf_dim*8, self.gf_dim*8, use_bn=use_bn)
        ''' siamese down end'''

        # input 60*60*1
        self.up_sample1 = up_sample(self.gf_dim*8, self.gf_dim*8, use_bn=use_bn)        # it will cat in next step
        self.up_sample2 = up_sample(self.gf_dim*8*2, self.gf_dim*8, use_bn=use_bn)
        self.up_sample3 = up_sample(self.gf_dim*8*2, self.gf_dim*8, use_bn=use_bn)
        self.up_sample4 = up_sample(self.gf_dim*8*2, self.gf_dim*8, use_bn=use_bn)
        self.up_sample5 = up_sample(self.gf_dim*8*2, self.gf_dim*4, use_bn=use_bn)
        self.up_sample6 = up_sample(self.gf_dim*8, self.gf_dim*2, use_bn=use_bn)
        self.up_sample7 = up_sample(self.gf_dim*4, self.gf_dim, use_bn=use_bn)
        self.up_sample8 = up_sample(self.gf_dim*2, output_nc, use_bn=use_bn)

    def forward(self, x):                                               # 1x3x256x256
        down1_skeleton_out = self.down1_skeleton(x)                     # 1x64x128x128
        down2_skeleton_out = self.down2_skeleton(down1_skeleton_out)    # 1x128x64x64
        down3_skeleton_out = self.down3_skeleton(down2_skeleton_out)    # 1x256x32x32
        down4_skeleton_out = self.down4_skeleton(down3_skeleton_out)    # 1x512x16x16
        down5_skeleton_out = self.down5_skeleton(down4_skeleton_out)    # 1x512x8x8
        down6_skeleton_out = self.down6_skeleton(down5_skeleton_out)    # 1x512x4x4
        down7_skeleton_out = self.down7_skeleton(down6_skeleton_out)    # 1x512x2x2
        down8_skeleton_out = self.down8_skeleton(down7_skeleton_out)    # 1x512x1x1

        down1_real_out = self.down1_real(x)
        down2_real_out = self.down2_real(down1_real_out)
        down3_real_out = self.down3_real(down2_real_out)
        down4_real_out = self.down4_real(down3_real_out)
        down5_real_out = self.down5_real(down4_real_out)
        down6_real_out = self.down6_real(down5_real_out)
        down7_real_out = self.down7_real(down6_real_out)
        down8_real_out = self.down8_real(down7_real_out)

        # mid
        down8_skeleton_real_out = torch.cat((down8_skeleton_out, down8_real_out), 2)        # 1x512x2x1

        up_sample_1 = self.up_sample1(down8_skeleton_real_out)                              # 1x512x4x2
        up_skeleton_real_out_1 = torch.cat((down7_skeleton_out, down7_real_out), 2)         # 1x512x4x2
        up_out_1 = torch.cat((up_sample_1, up_skeleton_real_out_1), 1)                      # 1x1024x4x2

        up_sample_2 = self.up_sample2(up_out_1)                                             # 1x512x8x4
        up_skeleton_real_out_2 = torch.cat((down6_skeleton_out, down6_real_out), 2)         # 1x512x8x4
        up_out_2 = torch.cat((up_sample_2, up_skeleton_real_out_2), 1)                      # 1x1024x8x4

        up_sample_3 = self.up_sample3(up_out_2)                                             # 1x512x16x8
        up_skeleton_real_out_3 = torch.cat((down5_skeleton_out, down5_real_out), 2)         # 1x512x16x8
        up_out_3 = torch.cat((up_sample_3, up_skeleton_real_out_3), 1)                      # 1x1024x16x8

        up_sample_4 = self.up_sample4(up_out_3)                                             # 1x512x32x16
        up_skeleton_real_out_4 = torch.cat((down4_skeleton_out, down4_real_out), 2)         # 1x512x32x16
        up_out_4 = torch.cat((up_sample_4, up_skeleton_real_out_4), 1)                      # 1x1024x32x16

        up_sample_5 = self.up_sample5(up_out_4)                                             # 1x256x64x32
        up_skeleton_real_out_5 = torch.cat((down3_skeleton_out, down3_real_out), 2)         # 1x256x64x32
        up_out_5 = torch.cat((up_sample_5, up_skeleton_real_out_5), 1)                      # 1x512x64x32

        up_sample_6 = self.up_sample6(up_out_5)                                             # 1x128x128x64
        up_skeleton_real_out_6 = torch.cat((down2_skeleton_out, down2_real_out), 2)         # 1x128x128x64
        up_out_6 = torch.cat((up_sample_6, up_skeleton_real_out_6), 1)                      # 1x256x128x64

        up_sample_7 = self.up_sample7(up_out_6)                                             # 1x64x256x128
        up_skeleton_real_out_7 = torch.cat((down1_skeleton_out, down1_real_out), 2)         # 1x64x256x128
        up_out_7 = torch.cat((up_sample_7, up_skeleton_real_out_7), 1)                      # 1x128x256x128

        up_sample_8 = self.up_sample8(up_out_7)                                             # 1x3x512x256
        # 256 * 512 乘 512 * 256 可以变形?

        return up_sample_8


class siamese_Unet_2(nn.Module):
    def __init__(self, input_nc, output_nc, use_bn=False):
        super(siamese_Unet_2, self).__init__()
        self.gf_dim = 64

        ''' siamese down start'''
        # input 256*256*3
        self.down1_skeleton = down_sample(input_nc, self.gf_dim, use_bn=False)
        self.down2_skeleton = down_sample(self.gf_dim, self.gf_dim * 2, use_bn=use_bn)
        self.down3_skeleton = down_sample(self.gf_dim * 2, self.gf_dim * 4, use_bn=use_bn)
        self.down4_skeleton = down_sample(self.gf_dim * 4, self.gf_dim * 8, use_bn=use_bn)
        self.down5_skeleton = down_sample(self.gf_dim * 8, self.gf_dim * 8, use_bn=use_bn)
        self.down6_skeleton = down_sample(self.gf_dim * 8, self.gf_dim * 8, use_bn=use_bn)
        self.down7_skeleton = down_sample(self.gf_dim * 8, self.gf_dim * 8, use_bn=use_bn)
        self.down8_skeleton = down_sample(self.gf_dim * 8, self.gf_dim * 8, use_bn=use_bn)
        # input 256*256*3
        self.down1_real = down_sample(input_nc, self.gf_dim, use_bn=False)
        self.down2_real = down_sample(self.gf_dim, self.gf_dim * 2, use_bn=use_bn)
        self.down3_real = down_sample(self.gf_dim * 2, self.gf_dim * 4, use_bn=use_bn)
        self.down4_real = down_sample(self.gf_dim * 4, self.gf_dim * 8, use_bn=use_bn)
        self.down5_real = down_sample(self.gf_dim * 8, self.gf_dim * 8, use_bn=use_bn)
        self.down6_real = down_sample(self.gf_dim * 8, self.gf_dim * 8, use_bn=use_bn)
        self.down7_real = down_sample(self.gf_dim * 8, self.gf_dim * 8, use_bn=use_bn)
        self.down8_real = down_sample(self.gf_dim * 8, self.gf_dim * 8, use_bn=use_bn)
        ''' siamese down end'''

        # cd512-cd1024-cd1024-c1024-c1024-c512-c256-c128 看第2位参数
        self.up_sample1 = up_sample(self.gf_dim * 8 * 2, self.gf_dim * 8 * 2, use_bn=use_bn)  # it will cat in next step
        self.up_sample2 = up_sample(self.gf_dim * 8 * 2 * 2, self.gf_dim * 8 * 2, use_bn=use_bn)
        self.up_sample3 = up_sample(self.gf_dim * 8 * 2 * 2, self.gf_dim * 8 * 2, use_bn=use_bn)
        self.up_sample4 = up_sample(self.gf_dim * 8 * 2 * 2, self.gf_dim * 8 * 2, use_bn=use_bn)
        self.up_sample5 = up_sample(self.gf_dim * 8 * 2 * 2, self.gf_dim * 8, use_bn=use_bn)
        self.up_sample6 = up_sample(self.gf_dim * 8 * 2, self.gf_dim * 4, use_bn=use_bn)
        self.up_sample7 = up_sample(self.gf_dim * 8, self.gf_dim * 2, use_bn=use_bn)
        self.up_sample8 = up_sample(self.gf_dim * 4, output_nc, use_bn=use_bn)

        self.tanh = nn.Tanh()

    def forward(self, x1, x2):  # 1x3x256x256
        down1_skeleton_out = self.down1_skeleton(x1)  # 1x64x128x128
        down2_skeleton_out = self.down2_skeleton(down1_skeleton_out)  # 1x128x64x64
        down3_skeleton_out = self.down3_skeleton(down2_skeleton_out)  # 1x256x32x32
        down4_skeleton_out = self.down4_skeleton(down3_skeleton_out)  # 1x512x16x16
        down5_skeleton_out = self.down5_skeleton(down4_skeleton_out)  # 1x512x8x8
        down6_skeleton_out = self.down6_skeleton(down5_skeleton_out)  # 1x512x4x4
        down7_skeleton_out = self.down7_skeleton(down6_skeleton_out)  # 1x512x2x2
        down8_skeleton_out = self.down8_skeleton(down7_skeleton_out)  # 1x512x1x1

        down1_real_out = self.down1_real(x2)
        down2_real_out = self.down2_real(down1_real_out)
        down3_real_out = self.down3_real(down2_real_out)
        down4_real_out = self.down4_real(down3_real_out)
        down5_real_out = self.down5_real(down4_real_out)
        down6_real_out = self.down6_real(down5_real_out)
        down7_real_out = self.down7_real(down6_real_out)
        down8_real_out = self.down8_real(down7_real_out)

        # mid
        # means decoder layer 1
        down8_skeleton_real_out = torch.cat((down8_skeleton_out, down8_real_out), 1)    # 1x1024x1x1

        up_sample_1 = self.up_sample1(down8_skeleton_real_out)                          # 1x1024x2x2
        up_skeleton_real_out_1 = torch.cat((down7_skeleton_out, down7_real_out), 1)     # 1x1024x2x2
        up_out_1 = torch.cat((up_sample_1, up_skeleton_real_out_1), 1)                  # 1x2048x2x2

        up_sample_2 = self.up_sample2(up_out_1)                                         # 1x1024x4x4
        up_skeleton_real_out_2 = torch.cat((down6_skeleton_out, down6_real_out), 1)     # 1x1024x4x4
        up_out_2 = torch.cat((up_sample_2, up_skeleton_real_out_2), 1)                  # 1x2048x8x4

        up_sample_3 = self.up_sample3(up_out_2)                                         # 1x1024x8x8
        up_skeleton_real_out_3 = torch.cat((down5_skeleton_out, down5_real_out), 1)     # 1x1024x8x8
        up_out_3 = torch.cat((up_sample_3, up_skeleton_real_out_3), 1)                  # 1x2048x8x8

        up_sample_4 = self.up_sample4(up_out_3)                                         # 1x1024x16x16
        up_skeleton_real_out_4 = torch.cat((down4_skeleton_out, down4_real_out), 1)     # 1x1024x16x16
        up_out_4 = torch.cat((up_sample_4, up_skeleton_real_out_4), 1)                  # 1x2048x16x16

        up_sample_5 = self.up_sample5(up_out_4)                                         # 1x512x32x32
        up_skeleton_real_out_5 = torch.cat((down3_skeleton_out, down3_real_out), 1)     # 1x512x32x32
        up_out_5 = torch.cat((up_sample_5, up_skeleton_real_out_5), 1)                  # 1x1024x32x32

        up_sample_6 = self.up_sample6(up_out_5)                                         # 1x256x64x64
        up_skeleton_real_out_6 = torch.cat((down2_skeleton_out, down2_real_out), 1)     # 1x256x64x64
        up_out_6 = torch.cat((up_sample_6, up_skeleton_real_out_6), 1)                  # 1x512x64x64

        up_sample_7 = self.up_sample7(up_out_6)                                         # 1x128x128x128
        up_skeleton_real_out_7 = torch.cat((down1_skeleton_out, down1_real_out), 1)     # 1x128x128x128
        up_out_7 = torch.cat((up_sample_7, up_skeleton_real_out_7), 1)                  # 1x256x128x128

        up_sample_8 = self.up_sample8(up_out_7)                                         # 1x3x256x256

        #return self.tanh(up_sample_8)
        return up_sample_8


class siamese_Unet_3(nn.Module):
    def __init__(self, input_nc, output_nc, use_bn=False):
        super(siamese_Unet_3, self).__init__()
        self.gf_dim = 64

        # input 256*256*3
        #self.down1_real = down_sample(input_nc, self.gf_dim, use_bn=False)
        self.down1_real = nn.Conv2d(input_nc, self.gf_dim, kernel_size=4, stride=2, padding=1, bias=True)
        self.down2_real = down_sample(self.gf_dim, self.gf_dim * 2, use_bn=use_bn)
        self.down3_real = down_sample(self.gf_dim * 2, self.gf_dim * 4, use_bn=use_bn)
        self.down4_real = down_sample(self.gf_dim * 4, self.gf_dim * 8, use_bn=use_bn)
        self.down5_real = down_sample(self.gf_dim * 8, self.gf_dim * 8, use_bn=use_bn)
        self.down6_real = down_sample(self.gf_dim * 8, self.gf_dim * 8, use_bn=use_bn)
        self.down7_real = down_sample(self.gf_dim * 8, self.gf_dim * 8, use_bn=use_bn)
        self.down8_real = down_sample(self.gf_dim * 8, self.gf_dim * 8, use_bn=False)
        ''' siamese down end'''

        # cd512-cd1024-cd1024-c1024-c1024-c512-c256-c128 看第2位参数
        self.up_sample1 = up_sample(self.gf_dim * 8, self.gf_dim * 8, dropout=False)  # it will cat in next step
        #self.up_sample1 = up_sample_start(self.gf_dim * 8, self.gf_dim * 8)  # it will cat in next step
        self.up_sample2 = up_sample(self.gf_dim * 8 * 2 , self.gf_dim * 8, dropout=True)
        self.up_sample3 = up_sample(self.gf_dim * 8 * 2, self.gf_dim * 8, dropout=True)
        self.up_sample4 = up_sample(self.gf_dim * 8 * 2, self.gf_dim * 8, dropout=True)
        self.up_sample5 = up_sample(self.gf_dim * 8 * 2, self.gf_dim * 4, dropout=False)
        self.up_sample6 = up_sample(self.gf_dim * 8, self.gf_dim * 2, dropout=False)
        self.up_sample7 = up_sample(self.gf_dim * 4, self.gf_dim, dropout=False)
        self.up_sample8 = up_sample_result(self.gf_dim * 2, output_nc)

    def forward(self, x):                                      # 1x3x256x256
        down1_real_out = self.down1_real(x)                        # 1x64x128x128
        down2_real_out = self.down2_real(down1_real_out)            # 1x128x64x64
        down3_real_out = self.down3_real(down2_real_out)            # 1x256x32x32
        down4_real_out = self.down4_real(down3_real_out)            # 1x512x16x16
        down5_real_out = self.down5_real(down4_real_out)            # 1x512x8x8
        down6_real_out = self.down6_real(down5_real_out)            # 1x512x4x4
        down7_real_out = self.down7_real(down6_real_out)            # 1x512x2x2
        down8_real_out = self.down8_real(down7_real_out)            # 1x512x1x1

        up_sample_1 = self.up_sample1(down8_real_out)               # 1x512x2x2
        up_out_1 = torch.cat((up_sample_1, down7_real_out), 1)      # 1x1024x2x2

        up_sample_2 = self.up_sample2(up_out_1)
        up_out_2 = torch.cat((up_sample_2, down6_real_out), 1)      # 1x1024x2x2

        up_sample_3 = self.up_sample3(up_out_2)
        up_out_3 = torch.cat((up_sample_3, down5_real_out), 1)

        up_sample_4 = self.up_sample4(up_out_3)
        up_out_4 = torch.cat((up_sample_4, down4_real_out), 1)

        up_sample_5 = self.up_sample5(up_out_4)
        up_out_5 = torch.cat((up_sample_5, down3_real_out), 1)

        up_sample_6 = self.up_sample6(up_out_5)
        up_out_6 = torch.cat((up_sample_6, down2_real_out), 1)

        up_sample_7 = self.up_sample7(up_out_6)
        up_out_7 = torch.cat((up_sample_7, down1_real_out), 1)

        up_sample_8 = self.up_sample8(up_out_7)

        return up_sample_8


class siamese_Unet_4(nn.Module):
    def __init__(self, input_nc, output_nc, use_bn=False):
        super(siamese_Unet_4, self).__init__()
        self.gf_dim = 64

        ''' siamese down start'''
        # input 256*256*3
        #self.down1_skeleton = down_sample(input_nc, self.gf_dim, use_bn=False)
        self.down1_skeleton = nn.Conv2d(input_nc, self.gf_dim, kernel_size=4, stride=2, padding=1, bias=True)
        self.down2_skeleton = down_sample(self.gf_dim, self.gf_dim * 2, use_bn=use_bn)
        self.down3_skeleton = down_sample(self.gf_dim * 2, self.gf_dim * 4, use_bn=use_bn)
        self.down4_skeleton = down_sample(self.gf_dim * 4, self.gf_dim * 8, use_bn=use_bn)
        self.down5_skeleton = down_sample(self.gf_dim * 8, self.gf_dim * 8, use_bn=use_bn)
        self.down6_skeleton = down_sample(self.gf_dim * 8, self.gf_dim * 8, use_bn=use_bn)
        self.down7_skeleton = down_sample(self.gf_dim * 8, self.gf_dim * 8, use_bn=use_bn)
        self.down8_skeleton = down_sample(self.gf_dim * 8, self.gf_dim * 8, use_bn=False)
        # input 256*256*3
        #self.down1_real = down_sample(input_nc, self.gf_dim, use_bn=False)
        self.down1_real = nn.Conv2d(input_nc, self.gf_dim, kernel_size=4, stride=2, padding=1, bias=True)
        self.down2_real = down_sample(self.gf_dim, self.gf_dim * 2, use_bn=use_bn)
        self.down3_real = down_sample(self.gf_dim * 2, self.gf_dim * 4, use_bn=use_bn)
        self.down4_real = down_sample(self.gf_dim * 4, self.gf_dim * 8, use_bn=use_bn)
        self.down5_real = down_sample(self.gf_dim * 8, self.gf_dim * 8, use_bn=use_bn)
        self.down6_real = down_sample(self.gf_dim * 8, self.gf_dim * 8, use_bn=use_bn)
        self.down7_real = down_sample(self.gf_dim * 8, self.gf_dim * 8, use_bn=use_bn)
        self.down8_real = down_sample(self.gf_dim * 8, self.gf_dim * 8, use_bn=False)
        ''' siamese down end'''

        # cd512-cd1024-cd1024-c1024-c1024-c512-c256-c128 看第2位参数
        self.up_sample1 = up_sample(self.gf_dim * 8 * 2, self.gf_dim * 8 * 2, dropout=False)  # it will cat in next step
        self.up_sample2 = up_sample(self.gf_dim * 8 * 2 * 2, self.gf_dim * 8 * 2, dropout=True)
        self.up_sample3 = up_sample(self.gf_dim * 8 * 2 * 2, self.gf_dim * 8 * 2, dropout=True)
        self.up_sample4 = up_sample(self.gf_dim * 8 * 2 * 2, self.gf_dim * 8 * 2, dropout=True)
        self.up_sample5 = up_sample(self.gf_dim * 8 * 2 * 2, self.gf_dim * 8, dropout=False)
        self.up_sample6 = up_sample(self.gf_dim * 8 * 2, self.gf_dim * 4, dropout=False)
        self.up_sample7 = up_sample(self.gf_dim * 8, self.gf_dim * 2, dropout=False)
        #self.up_sample8 = up_sample(self.gf_dim * 4, output_nc, use_bn=use_bn)
        self.up_sample8 = up_sample_result(self.gf_dim * 4, output_nc)

        self.tanh = nn.Tanh()

    def forward(self, x1, x2):  # 1x3x256x256
        down1_skeleton_out = self.down1_skeleton(x1)  # 1x64x128x128
        down2_skeleton_out = self.down2_skeleton(down1_skeleton_out)  # 1x128x64x64
        down3_skeleton_out = self.down3_skeleton(down2_skeleton_out)  # 1x256x32x32
        down4_skeleton_out = self.down4_skeleton(down3_skeleton_out)  # 1x512x16x16
        down5_skeleton_out = self.down5_skeleton(down4_skeleton_out)  # 1x512x8x8
        down6_skeleton_out = self.down6_skeleton(down5_skeleton_out)  # 1x512x4x4
        down7_skeleton_out = self.down7_skeleton(down6_skeleton_out)  # 1x512x2x2
        down8_skeleton_out = self.down8_skeleton(down7_skeleton_out)  # 1x512x1x1

        down1_real_out = self.down1_real(x2)
        down2_real_out = self.down2_real(down1_real_out)
        down3_real_out = self.down3_real(down2_real_out)
        down4_real_out = self.down4_real(down3_real_out)
        down5_real_out = self.down5_real(down4_real_out)
        down6_real_out = self.down6_real(down5_real_out)
        down7_real_out = self.down7_real(down6_real_out)
        down8_real_out = self.down8_real(down7_real_out)

        # mid
        # means decoder layer 1
        down8_skeleton_real_out = torch.cat((down8_skeleton_out, down8_real_out), 1)    # 1x1024x1x1

        up_sample_1 = self.up_sample1(down8_skeleton_real_out)                          # 1x1024x2x2
        up_skeleton_real_out_1 = torch.cat((down7_skeleton_out, down7_real_out), 1)     # 1x1024x2x2
        up_out_1 = torch.cat((up_sample_1, up_skeleton_real_out_1), 1)                  # 1x2048x2x2

        up_sample_2 = self.up_sample2(up_out_1)                                         # 1x1024x4x4
        up_skeleton_real_out_2 = torch.cat((down6_skeleton_out, down6_real_out), 1)     # 1x1024x4x4
        up_out_2 = torch.cat((up_sample_2, up_skeleton_real_out_2), 1)                  # 1x2048x8x4

        up_sample_3 = self.up_sample3(up_out_2)                                         # 1x1024x8x8
        up_skeleton_real_out_3 = torch.cat((down5_skeleton_out, down5_real_out), 1)     # 1x1024x8x8
        up_out_3 = torch.cat((up_sample_3, up_skeleton_real_out_3), 1)                  # 1x2048x8x8

        up_sample_4 = self.up_sample4(up_out_3)                                         # 1x1024x16x16
        up_skeleton_real_out_4 = torch.cat((down4_skeleton_out, down4_real_out), 1)     # 1x1024x16x16
        up_out_4 = torch.cat((up_sample_4, up_skeleton_real_out_4), 1)                  # 1x2048x16x16

        up_sample_5 = self.up_sample5(up_out_4)                                         # 1x512x32x32
        up_skeleton_real_out_5 = torch.cat((down3_skeleton_out, down3_real_out), 1)     # 1x512x32x32
        up_out_5 = torch.cat((up_sample_5, up_skeleton_real_out_5), 1)                  # 1x1024x32x32

        up_sample_6 = self.up_sample6(up_out_5)                                         # 1x256x64x64
        up_skeleton_real_out_6 = torch.cat((down2_skeleton_out, down2_real_out), 1)     # 1x256x64x64
        up_out_6 = torch.cat((up_sample_6, up_skeleton_real_out_6), 1)                  # 1x512x64x64

        up_sample_7 = self.up_sample7(up_out_6)                                         # 1x128x128x128
        up_skeleton_real_out_7 = torch.cat((down1_skeleton_out, down1_real_out), 1)     # 1x128x128x128
        up_out_7 = torch.cat((up_sample_7, up_skeleton_real_out_7), 1)                  # 1x256x128x128

        up_sample_8 = self.up_sample8(up_out_7)                                         # 1x3x256x256

        return up_sample_8

