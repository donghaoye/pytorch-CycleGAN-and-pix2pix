#coding=utf8
import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import itertools
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import sys

class CycleGANAABBModel(BaseModel):
    def name(self):
        return 'CycleGANAABBModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        nb = opt.batchSize
        size = opt.fineSize

        self.input_A1 = self.Tensor(nb, opt.input_nc, size, size)
        self.input_A2 = self.Tensor(nb, opt.input_nc, size, size)
        self.input_B1 = self.Tensor(nb, opt.output_nc, size, size)
        self.input_B2 = self.Tensor(nb, opt.output_nc, size, size)

        # load/define networks
        # The naming conversion is different from those used in the paper
        # Code (paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.which_model_netG, opt.norm, opt.use_dropout, self.gpu_ids)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.which_model_netG, opt.norm, opt.use_dropout, self.gpu_ids)
        self.netG_A.train()
        self.netG_B.train()

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD_A = networks.define_D(opt.input_nc + opt.input_nc + opt.output_nc, opt.ndf, opt.which_model_netD, opt.n_layers_D, opt.norm, use_sigmoid, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc + opt.input_nc + opt.output_nc, opt.ndf, opt.which_model_netD, opt.n_layers_D, opt.norm, use_sigmoid, self.gpu_ids)

        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netG_A, 'G_A', which_epoch)
            self.load_network(self.netG_B, 'G_B', which_epoch)
            if self.isTrain:
                self.load_network(self.netD_A, 'D_A', which_epoch)
                self.load_network(self.netD_B, 'D_B', which_epoch)

        if self.isTrain:
            self.old_lr = opt.lr
            self.fake_A1_pool = ImagePool(opt.pool_size)
            self.fake_A2_pool = ImagePool(opt.pool_size)
            self.fake_B1_pool = ImagePool(opt.pool_size)
            self.fake_B2_pool = ImagePool(opt.pool_size)
            self.fake_AB_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            self.criterionPOSE = networks.POSELoss()
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers
            #self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_G = torch.optim.Adam(self.netG_A.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

            print('---------- Networks initialized -------------')
            networks.print_network(self.netG_A)
            networks.print_network(self.netG_B)
            networks.print_network(self.netD_A)
            networks.print_network(self.netD_B)
            print('-----------------------------------------------')

    def set_input(self, input):
        input_A1 = input['A1']
        input_A2 = input['A2']
        input_B1 = input['B1']
        input_B2 = input['B2']
        self.input_A1.resize_(input_A1.size()).copy_(input_A1)
        self.input_A2.resize_(input_A2.size()).copy_(input_A2)
        self.input_B1.resize_(input_B1.size()).copy_(input_B1)
        self.input_B2.resize_(input_B2.size()).copy_(input_B2)
        self.image_path_A1 = input['A1_paths']
        self.image_path_A2 = input['A2_paths']
        self.image_path_B1 = input['B1_paths']
        self.image_path_B2 = input['B2_paths']

    def forward(self):
        self.real_A1 = Variable(self.input_A1)
        self.real_A2 = Variable(self.input_A2)
        self.real_B1 = Variable(self.input_B1)
        self.real_B2 = Variable(self.input_B2)
        self.fake_B1 = self.netG_A.train().forward(self.real_A1, self.real_B2)

    def test(self):
        self.real_A1 = Variable(self.input_A1, volatile=True)
        self.real_A2 = Variable(self.input_A2, volatile=True)
        self.real_B1 = Variable(self.input_B1, volatile=True)
        self.real_B2 = Variable(self.input_B2, volatile=True)

        self.fake_B1 = self.netG_A.forward(self.real_A1, self.real_B2)
        self.rec_A1 = self.netG_B.forward(self.fake_B1, self.real_A2)
        self.fake_A1 = self.netG_B.forward(self.real_B1, self.real_A2)
        self.rec_B2  = self.netG_A.forward(self.fake_A1, self.real_B2)

    #get image paths
    def get_image_paths(self):
        return {"A1": self.image_path_A1, "A2": self.image_path_A2, "B1": self.image_path_B1, "B2": self.image_path_B2}

    #  送入D还是一张图片
    def backward_D_basic(self, netD, real_A1, skeleton, real, fake):

        # Fake
        fake_AB = self.fake_AB_pool.query(torch.cat((real_A1, skeleton, fake), 1))
        pred_fake = netD.forward(fake_AB.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)

        # Real
        real_AB = torch.cat((real_A1, skeleton, real), 1)
        pred_real = netD.forward(real_AB)
        loss_D_real = self.criterionGAN(pred_real, True)


        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        #fake_B1 = self.fake_B1_pool.query(self.fake_B1)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_A1, self.real_B2, self.real_B1, self.fake_B1)

    def backward_D_B(self):
        #fake_A1 = self.fake_A1_pool.query(self.fake_A1)
        self.loss_D_B =  self.backward_D_basic(self.netD_B, self.real_B1, self.real_A2, self.real_A1, self.fake_A1)

    def backward_G(self):
        lambda_idt = self.opt.identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity loss 这里默认为0.0, 没有用上
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed.
            self.fake_A1 = self.netG_A.forward(self.real_B1, self.real_A2)
            self.loss_idt_A = self.criterionIdt(self.idt_A1, self.real_A1) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed.
            self.idt_B1 = self.netG_B.forward(self.real_A1, self.real_B2)
            self.loss_idt_B = self.criterionIdt(self.idt_B1, self.real_B1) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0


        # GAN loss
        # D_A(G_A(A))
        #self.fake_B1 = self.netG_A.forward(self.real_A1, self.real_B2)
        fake_AB = torch.cat((self.real_A1, self.real_B2, self.fake_B1), 1)
        pred_fake = self.netD_A.forward(fake_AB)
        self.loss_G_A = self.criterionGAN(pred_fake, True)
        # D_B(G_B(B))
        # self.fake_A1 = self.netG_B.forward(self.real_B1, self.real_A2)
        # pred_fake = self.netD_B.forward(self.real_B1, self.real_A2, self.fake_A1)
        # self.loss_G_B = self.criterionGAN(pred_fake, True)

        # L1 loss
        self.loss_G_A_L1 = self.criterionL1(self.fake_B1, self.real_B1) * self.opt.lambda_A

        # rec loss 直接用 pose loss来的代替
        # Forward cycle loss
        #self.rec_A1 = self.netG_B.forward(self.fake_B1, self.real_A2)
        #self.loss_cycle_A = self.criterionCycle(self.rec_A1, self.real_A1) * lambda_A
        #self.loss_cycle_A = self.criterionPOSE(self.real_A1, self.fake_B1, self.real_B1) * 0
        # Backward cycle loss
        #self.rec_B1 = self.netG_A.forward(self.fake_A1, self.real_B2)
        #self.loss_cycle_B = self.criterionCycle(self.rec_B1, self.real_B1) * lambda_B
        #self.loss_cycle_B = self.criterionPOSE(self.real_B1, self.fake_A1, self.real_A1) * 0
        # combined loss
        #self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
        # self.loss_G = self.loss_G_A + self.loss_cycle_A + self.loss_idt_A
        self.loss_G = self.loss_G_A + self.loss_G_A_L1
        self.loss_G.backward()

    def optimize_parameters(self):
        # forward
        self.forward()
        # G_A and G_B
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        # D_A
        self.optimizer_D_A.zero_grad()
        self.backward_D_A()
        self.optimizer_D_A.step()
        # # D_B
        # self.optimizer_D_B.zero_grad()
        # self.backward_D_B()
        # self.optimizer_D_B.step()


    def get_current_errors(self):
        D_A = self.loss_D_A.data[0]
        G_A = self.loss_G_A.data[0]
        L1_A = self.loss_G_A_L1.data[0]

        return OrderedDict([('D_A', D_A), ('G_A', G_A), ('L1_A', L1_A)])
        # D_A = self.loss_D_A.data[0]
        # G_A = self.loss_G_A.data[0]
        # Cyc_A = self.loss_cycle_A.data[0]
        # D_B = self.loss_D_B.data[0]
        # G_B = self.loss_G_B.data[0]
        # Cyc_B = self.loss_cycle_B.data[0]
        # if self.opt.identity > 0.0:
        #     idt_A = self.loss_idt_A.data[0]
        #     idt_B = self.loss_idt_B.data[0]
        #     return OrderedDict([('D_A', D_A), ('G_A', G_A), ('Cyc_A', Cyc_A), ('idt_A', idt_A),
        #                          ('G_B', G_B), ('Cyc_B', Cyc_B), ('idt_B', idt_B),
        #                         ('D_B', D_B), ('G_B', G_B), ('Cyc_B', Cyc_B), ('idt_B', idt_B)])
        # else:
        #     return OrderedDict([('D_A', D_A), ('G_A', G_A), ('Cyc_A', Cyc_A),
        #                         ('G_B', G_B), ('Cyc_B', Cyc_B),
        #                        ('D_B', D_B), ('G_B', G_B), ('Cyc_B', Cyc_B)])

    def get_current_visuals(self):
        real_A1 = util.tensor2im(self.real_A1.data)
        real_A2 = util.tensor2im(self.real_A2.data)
        real_B1 = util.tensor2im(self.real_B1.data)
        real_B2 = util.tensor2im(self.real_B2.data)
        fake_B1 = util.tensor2im(self.fake_B1.data)

        return OrderedDict([('real_A1', real_A1), ('real_A2', real_A2),
                                ('real_B1', real_B1), ('real_B2', real_B2),
                                ('fake_B1', fake_B1)])

        # real_A1 = util.tensor2im(self.real_A1.data)
        # real_A2 = util.tensor2im(self.real_A2.data)
        # real_B1 = util.tensor2im(self.real_B1.data)
        # real_B2 = util.tensor2im(self.real_B2.data)
        # fake_A1 = util.tensor2im(self.fake_A1.data)
        # fake_B1 = util.tensor2im(self.fake_B1.data)
        # rec_A1  = util.tensor2im(self.rec_A1.data)
        # rec_B1  = util.tensor2im(self.rec_B1.data)
        # if self.opt.identity > 0.0:
        #     idt_A1 = util.tensor2im(self.idt_A1.data)
        #     idt_B1 = util.tensor2im(self.idt_B1.data)
        #     return OrderedDict([('real_A1', real_A1), ('real_A2', real_A2),
        #                         ('real_B1', real_B1), ('real_B2', real_B2),
        #                         ('fake_A1', fake_A1), ('fake_B1', fake_B1),
        #                         ('rec_A1', rec_A1),  ('rec_B1', rec_B1),
        #                         ('idt_A1', idt_A1), ('idt_B1', idt_B1)
        #                         ])
        # else:
        #     return OrderedDict([('real_A1', real_A1), ('real_A2', real_A2),
        #                         ('real_B1', real_B1), ('real_B2', real_B2),
        #                         ('fake_A1', fake_A1), ('fake_B1', fake_B1),
        #                         ('rec_A1', rec_A1), ('rec_B1', rec_B1)
        #                         ])

    def save(self, label):
        self.save_network(self.netG_A, 'G_A', label, self.gpu_ids)
        self.save_network(self.netD_A, 'D_A', label, self.gpu_ids)
        self.save_network(self.netG_B, 'G_B', label, self.gpu_ids)
        self.save_network(self.netD_B, 'D_B', label, self.gpu_ids)

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_D_A.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_D_B.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr

        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr
