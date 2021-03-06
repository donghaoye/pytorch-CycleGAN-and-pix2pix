#coding=utf8
import random
import torch.utils.data
import torchvision.transforms as transforms
from data.base_data_loader import BaseDataLoader
from data.image_folder import ImageFolder
# pip install future --upgrade
from builtins import object
#from pdb import set_trace as st

from PIL import Image, ImageChops, ImageOps

"""
(A,B,C)
这里是迭代器, Iterator
1）如果使用列表，计算值时会一次获取所有值，那么就会占用更多的内存。而迭代器则是一个接一个计算。
2）使代码更通用、更简单。
"""
class ThreePairedData(object):
    def __init__(self, data_loader_A, data_loader_B, data_loader_C, max_dataset_size, flip):
        self.data_loader_A = data_loader_A
        self.data_loader_B = data_loader_B
        self.data_loader_C = data_loader_C
        self.stop_A = False
        self.stop_B = False
        self.stop_C = False
        self.max_dataset_size = max_dataset_size
        self.flip = flip

    def __iter__(self):
        self.stop_A = False
        self.stop_B = False
        self.stop_C = False
        self.data_loader_A_iter = iter(self.data_loader_A)
        self.data_loader_B_iter = iter(self.data_loader_B)
        self.data_loader_C_iter = iter(self.data_loader_C)
        self.iter = 0
        return self

    def __next__(self):
        A, A_paths = None, None
        B, B_paths = None, None
        C, C_paths = None, None

        try:
            A, A_paths = next(self.data_loader_A_iter)
        except StopIteration:
            if A is None or A_paths is None:
                self.stop_A = True
                self.data_loader_A_iter = iter(self.data_loader_A)
                A, A_paths = next(self.data_loader_A_iter)

        try:
            B, B_paths = next(self.data_loader_B_iter)
        except StopIteration:
            if B is None or B_paths is None:
                self.stop_B = True
                self.data_loader_B_iter = iter(self.data_loader_B)
                B, B_paths = next(self.data_loader_B_iter)

        try:
            C, C_paths = next(self.data_loader_C_iter)
        except StopIteration:
            if C is None or C_paths is None:
                self.stop_C = True
                self.data_loader_C_iter = iter(self.data_loader_C)
                C, C_paths = next(self.data_loader_C_iter)

        if (self.stop_A and self.stop_B and self.stop_C) or self.iter > self.max_dataset_size:
            self.stop_A = False
            self.stop_B = False
            self.stop_C = False
            raise StopIteration()
        else:
            self.iter += 1
            if self.flip and random.random() < 0.5:
                idx = [i for i in range(A.size(3) - 1, -1, -1)]
                idx = torch.LongTensor(idx)
                A = A.index_select(3, idx)
                B = B.index_select(3, idx)
                C = C.index_select(3, idx)
            return {'A': A, 'A_paths': A_paths,
                    'B': B, 'B_paths': B_paths,
                    'C': C, 'C_paths': C_paths}

# 指定长宽不变形缩放，填充0
class MyScale(object):
    def __init__(self, size=(256,256), pad=False):
        self.size = size
        self.pad = pad

    def __call__(self, img):
        img.thumbnail(self.size, Image.ANTIALIAS)
        image_size = img.size
        if self.pad:
            thumb = img.crop((0, 0, self.size[0], self.size[1]))
            offset_x = max((self.size[0] - image_size[0]) / 2, 0)
            offset_y = max((self.size[1] - image_size[1]) / 2, 0)
            thumb = ImageChops.offset(thumb, offset_x, offset_y)
        else:
            thumb = ImageOps.fit(img, self.size, Image.ANTIALIAS, (0.5, 0.5))
        return thumb

class ThreeAlignedDataLoader(BaseDataLoader):
    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        transformations = [
                            #transforms.Scale(opt.loadSize),
                            MyScale(size=(256, 256), pad=True),
                            transforms.RandomCrop(opt.fineSize),
                            transforms.ToTensor(),
                           # this is wrong! because the fake samples are not normalized like this,
                           # still they are inferred on the same network,
                            transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))
                           #lambda x: (x - x.min()) / x.max() * 2 - 1,  # [-1., 1.]
                           ] # 归一化，会产生负数。

        #transformations = [transforms.Scale(opt.loadSize), transforms.RandomCrop(opt.fineSize),
        #                    transforms.ToTensor()]
        transform = transforms.Compose(transformations)

        # Dataset A, eg.. trainA目录
        dataset_A = ImageFolder(root=opt.dataroot + '/' + opt.phase + 'A', transform=transform, return_paths=True)
        data_loader_A = torch.utils.data.DataLoader(
            dataset_A, batch_size=self.opt.batchSize, shuffle=not self.opt.serial_batches, num_workers=int(self.opt.nThreads))

        # Dataset B
        dataset_B = ImageFolder(root=opt.dataroot + '/' + opt.phase + 'B', transform=transform, return_paths=True)
        data_loader_B = torch.utils.data.DataLoader(
            dataset_B, batch_size=self.opt.batchSize, shuffle=not self.opt.serial_batches, num_workers=int(self.opt.nThreads))

        # Dataset C
        dataset_C = ImageFolder(root=opt.dataroot + '/' + opt.phase + 'C', transform=transform, return_paths=True)
        data_loader_C = torch.utils.data.DataLoader(
            dataset_C, batch_size=self.opt.batchSize, shuffle=not self.opt.serial_batches, num_workers=int(self.opt.nThreads))

        # 如何保证 A、B、C是一一一对应的呢，shuffle=not self.opt.serial_batches 这个参数 serial_batches 为True，代表有序，否则随机
        # shuffle 是洗牌，搅乱的意思

        self.dataset_A = dataset_A
        self.dataset_B = dataset_B
        self.dataset_C = dataset_C
        flip = opt.isTrain and not opt.no_flip
        self.three_paired_data = ThreePairedData(data_loader_A, data_loader_B, data_loader_C, self.opt.max_dataset_size, flip)

    def name(self):
        return 'ThreeAlignedDataLoader'

    def load_data(self):
        return self.three_paired_data

    def __len__(self):
        return min(max(len(self.dataset_A), len(self.dataset_B), len(self.dataset_C)), self.opt.max_dataset_size)
