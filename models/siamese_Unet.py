#coding=utf8

import torch
import torch.nn as nn

def down_sample(dim_in, dim_out, kernel_size=4, stride=2, bias=True, use_bn=False):
    if use_bn:
        return nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, bias=bias),
            nn.BatchNorm2d(dim_out),
            nn.LeakyReLU(0.2)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, bias=bias),
            nn.LeakyReLU(0.2)
        )


def up_sample(dim_in, dim_out, kernel_size=4, stride=2, bias=False, use_bn=False):
    if use_bn:
        return nn.Sequential(
            nn.ConvTranspose2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, bias=bias),
            nn.BatchNorm2d(dim_out),
            nn.Dropout(0.5),
            nn.ReLU()
        )
    else:
        return nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, bias=bias),
            nn.Dropout(0.5),
            nn.ReLU()
        )

# def upsample(ch_coarse, ch_fine):
#   return nn.Sequential(
#     nn.ConvTranspose2d(ch_coarse, ch_fine, 4, 2, 1, bias=False),
#     nn.ReLU()
#   )

class siamese_Unet(nn.Module):
    def __init__(self, input_nc, output_nc, use_bn=False, gpu_ids=[]):
        super(siamese_Unet, self).__init__()

        ''' siamese down start'''
        self.down0_skeleton = down_sample(input_nc, 32, use_bn=False)
        self.down1_skeleton = down_sample(32, 64, use_bn=use_bn)
        self.down2_skeleton = down_sample(64, 128, use_bn=use_bn)
        self.down3_skeleton = down_sample(128, 256, use_bn=use_bn)
        self.down4_skeleton = down_sample(256, 512, use_bn=use_bn)
        # self.down5_skeleton = down_sample(256, 512, use_bn=use_bn)
        # self.down6_skeleton = down_sample(256, 512, use_bn=use_bn)
        # self.down7_skeleton = down_sample(256, 512, use_bn=use_bn)
        # self.down8_skeleton = down_sample(256, 512, use_bn=use_bn)

        self.down4_skeleton_m = down_sample(512, 256, use_bn=use_bn)
        self.down3_skeleton_m = down_sample(256, 128, use_bn=use_bn)
        self.down2_skeleton_m = down_sample(128, 64, use_bn=use_bn)
        self.down1_skeleton_m = down_sample(64, 32, use_bn=use_bn)

        self.down0_real = down_sample(1, 32, use_bn=False)
        self.down1_real = down_sample(32, 64, use_bn=use_bn)
        self.down2_real = down_sample(64, 128, use_bn=use_bn)
        self.down3_real = down_sample(128, 256, use_bn=use_bn)
        self.down4_real = down_sample(256, 512, use_bn=use_bn)
        # self.down5_real = down_sample(256, 512, use_bn=use_bn)
        # self.down6_real = down_sample(256, 512, use_bn=use_bn)
        # self.down7_real = down_sample(256, 512, use_bn=use_bn)
        # self.down8_real = down_sample(256, 512, use_bn=use_bn)

        self.down4_m = down_sample(512, 256, use_bn=use_bn)
        self.down3_m = down_sample(256, 128, use_bn=use_bn)
        self.down2_m = down_sample(128, 64, use_bn=use_bn)
        self.down1_m = down_sample(64, 32, use_bn=use_bn)

        ''' siamese down end'''

        self.up_sample43 = up_sample(512, 256)
        self.up_sample32 = up_sample(256, 128)
        self.up_sample21 = up_sample(128, 64)
        self.up_sample10 = up_sample(64, 32)


    def forward(self, x):
        x_len = len(x) / 2
        down0_skeleton_out = self.down0_skeleton(x[:x_len-1])
        down1_skeleton_out = self.down1_skeleton(down0_skeleton_out)
        down2_skeleton_out = self.down2_skeleton(down1_skeleton_out)
        down3_skeleton_out = self.down3_skeleton(down2_skeleton_out)
        down4_skeleton_out = self.down4_skeleton(down3_skeleton_out)
        # down5_skeleton_out = self.down5_skeleton(down4_skeleton_out)
        # down6_skeleton_out = self.down6_skeleton(down5_skeleton_out)
        # down7_skeleton_out = self.down7_skeleton(down6_skeleton_out)
        # down8_skeleton_out = self.down8_skeleton(down7_skeleton_out)

        down0_real_out = self.down0_real(x[x_len:])
        down1_real_out = self.down1_real(down0_real_out)
        down2_real_out = self.down2_real(down1_real_out)
        down3_real_out = self.down3_real(down2_real_out)
        down4_real_out = self.down4_real(down3_real_out)
        # down5_real_out = self.down5_real(down4_real_out)
        # down6_real_out = self.down6_real(down5_real_out)
        # down7_real_out = self.down7_real(down6_real_out)
        # down8_real_out = self.down8_real(down7_real_out)

        # mid
        down8_skeleton_real_out = torch.cat((down4_skeleton_out, down4_real_out), 0)

        up_sample_43 = self.up_sample43(down8_skeleton_real_out)
        up3_skeleton_real_out = torch.cat((down3_skeleton_out, down3_real_out), 0)
        up_out_43 = torch.cat((up_sample_43, up3_skeleton_real_out), 1)
        up_conv_out_43 = self.down4_skeleton_m(up_out_43)

        up_sample_32 = self.up_sample32(up_conv_out_43)
        up2_skeleton_real_out = torch.cat((down2_skeleton_out, down2_real_out), 0)
        up_out_32 = torch.cat((up_sample_32, up2_skeleton_real_out), 1)
        up_conv_out_32 = self.down3_skeleton_m(up_out_32)

        up_sample_21 = self.up_sample21(up_conv_out_32)
        up1_skeleton_real_out = torch.cat((down1_skeleton_out, down1_real_out), 0)
        up_out_21 = torch.cat((up_sample_21, up1_skeleton_real_out), 1)
        up_conv_out_21 = self.down2_skeleton_m(up_out_21)

        up_conv_out_10 = self.down1_skeleton_m(up_conv_out_21)

        return up_conv_out_10



