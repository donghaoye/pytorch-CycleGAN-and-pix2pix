import os
import torch
import torch.nn as nn
from torch.nn import MSELoss
from torch.utils.serialization import load_lua
from torch.autograd import Variable
import models
from data_utils import vgg_mean_subtraction

class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg16_model()
        self.vgg = models.VGGFeature()
        self.vgg.load_state_dict(torch.load('vgg16feature.pth'))
        self.criterion = torch.nn.MSELoss()


    def vgg16_model(self):
        if not os.path.exists('vgg16feature.pth'):
            if not os.path.exists('vgg16.t7'):
                os.system('wget http://cs.stanford.edu/people/jcjohns/fast-neural-style/models/vgg16.t7')
            vgglua = load_lua('vgg16.t7')
            vgg = models.VGGFeature()
            # for (src, dst) in zip(vgglua.parameters()[0], vgg.parameters()):
            #     dst[:] = src[:]
            torch.save(vgg.state_dict(), 'vgg16feature.pth')

    def gram_matrix(self, y):
        B, C, H, W = y.size()
        features = y.view(B, C, W*H)
        features_t = features.transpose(1,2)
        gram = features.bmm(features_t) / (C*H*W)
        return gram

    def __call__(self, content_weight, style_weight, yc, ys, y_hat, cuda):
    #def loss_function(content_weight, style_weight, yc, ys, y_hat, cuda):
        #self.vgg16_model()
        # vgg = models.VGGFeature()
        # vgg.load_state_dict(torch.load('vgg16feature.pth'))
        # criterion = torch.nn.MSELoss()

        if cuda:
            self.vgg = self.vgg.cuda()
            self.criterion = self.criterion.cuda()

        vgg_mean_subtraction(yc.clone())
        vgg_mean_subtraction(ys.clone())
        vgg_mean_subtraction(y_hat.clone())

        feature_c = self.vgg(yc.clone())
        feature_hat = self.vgg(y_hat.clone())
        feat_loss = content_weight * self.criterion(feature_hat[2], Variable(feature_c[2].data, requires_grad=False))

        feature_s = self.vgg(ys)
        gram_s = [self.gram_matrix(y) for y in feature_s]
        gram_hat = [self.gram_matrix(y) for y in feature_hat]
        style_loss = 0
        for m in range(0, len(feature_hat)):
            style_loss += style_weight * self.criterion(gram_hat[m], Variable(gram_s[m].data, requires_grad=False))

        return style_loss + feat_loss
