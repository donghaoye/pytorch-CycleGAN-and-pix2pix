#coding=utf8
import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks

class Pix2PixModelABC(BaseModel):
    def name(self):
        return 'Pix2PixModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        # define tensors
        self.input_A_1 = self.Tensor(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize)
        self.input_A_2 = self.Tensor(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize)
        self.input_B = self.Tensor(opt.batchSize, opt.output_nc, opt.fineSize, opt.fineSize)

        # load/define networks
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.which_model_netG, opt.norm, opt.use_dropout, self.gpu_ids)
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            # D 的输入 是 9 channel了
            self.netD = networks.define_D(opt.input_nc + opt.input_nc + opt.output_nc, opt.ndf,
                                         opt.which_model_netD, opt.n_layers_D, opt.norm, use_sigmoid, self.gpu_ids)
        if not self.isTrain or opt.continue_train:
            self.load_network(self.netG, 'G', opt.which_epoch)
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch)

        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionL1 = torch.nn.L1Loss()

            #self.triLoss = networks.TRILoss(tensor=self.Tensor)

            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

            print('---------- Networks initialized ------------')
            networks.print_network(self.netG)
            networks.print_network(self.netD)
            print('----------------------------------------------')

    # 这里传入的input是字典
    #        {'A': A, 'A_paths': A_paths,
    #         'B': B, 'B_paths': B_paths,
    #         'C': C, 'C_paths': C_paths}
    def set_input(self, input):
        #AtoB = self.opt.which_direction == 'AtoB'
        input_A_1 = input['A']
        input_A_2 = input['B']
        input_B   = input['C']
        # print input_A_1
        # print input_A_2
        # print input_B

        self.input_A_1.resize_(input_A_1.size()).copy_(input_A_1)
        self.input_A_2.resize_(input_A_2.size()).copy_(input_A_2)
        self.input_B.resize_(input_B.size()).copy_(input_B)
        self.image_path_A_1 = input['A_paths']
        self.image_path_A_2 = input['B_paths']
        self.image_path_B = input['C_paths']

        print self.image_path_A_1
        print self.image_path_A_2
        print self.image_path_B


    def forward(self):
        self.real_A_1 = Variable(self.input_A_1, volatile=True)
        self.real_A_2 = Variable(self.input_A_2, volatile=True)
        self.fake_B = self.netG.forward(self.real_A_1, self.real_A_2)
        self.real_B = Variable(self.input_B)

    # no backprop gradients
    def test(self):
        self.real_A_1 = Variable(self.input_A_1, volatile=True)
        self.real_A_2 = Variable(self.input_A_2, volatile=True)
        self.fake_B = self.netG.forward(self.real_A_1, self.real_A_2)
        self.real_B = Variable(self.input_B, volatile=True)

    #get image paths
    def get_image_paths(self):
        return self.image_path_A_1 + "," + self.image_path_A_2 + "," + self.image_path_B

    def backward_D(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        fake_AB = self.fake_AB_pool.query(torch.cat((self.real_A_1, self.real_A_2, self.fake_B), 1))
        self.pred_fake = self.netD.forward(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(self.pred_fake, False)

        # Real
        real_AB = torch.cat((self.real_A_1, self.real_A_2, self.real_B), 1)         #.detach()
        self.pred_real = self.netD.forward(real_AB)
        self.loss_D_real = self.criterionGAN(self.pred_real, True)

        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        # print fake_AB
        # print real_AB
        # print loss_D

        self.loss_D.backward()

    def backward_G(self):
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A_1, self.real_A_2, self.fake_B), 1)
        pred_fake = self.netD.forward(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        # Second, G(A) = B, 直接计算 G(A) 和 B 的L1距离（各坐标差的绝对值和）
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_A

        # tri loss
        #self.loss_G_tri = self.triLoss(self.fake_B, self.real_B, )

        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        #self.loss_G = self.loss_G_GAN

        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()

        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def get_current_errors(self):
        return OrderedDict([
                ('G_GAN',  self.loss_G_GAN.data[0]),
                ('G_L1',   self.loss_G_L1.data[0]),
                ('D_real', self.loss_D_real.data[0]),
                ('D_fake', self.loss_D_fake.data[0])
        ])

    def get_current_visuals(self):
        real_A_1 = util.tensor2im(self.real_A_1.data)
        real_A_2 = util.tensor2im(self.real_A_2.data)
        fake_B = util.tensor2im(self.fake_B.data)
        real_B = util.tensor2im(self.real_B.data)
        return OrderedDict([
            ('real_A_1', real_A_1),
            ('real_A_2', real_A_2),
            ('fake_B',   fake_B),
            ('real_B',   real_B)])

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        self.save_network(self.netD, 'D', label, self.gpu_ids)

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr