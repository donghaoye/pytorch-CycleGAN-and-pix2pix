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
class FourPairedData(object):
    def __init__(self, data_loader_A1, data_loader_A2, data_loader_B1, data_loader_B2, max_dataset_size, flip):
        self.data_loader_A1 = data_loader_A1
        self.data_loader_A2 = data_loader_A2
        self.data_loader_B1 = data_loader_B1
        self.data_loader_B2 = data_loader_B2
        self.stop_A1 = False
        self.stop_A2 = False
        self.stop_B1 = False
        self.stop_B2 = False
        self.max_dataset_size = max_dataset_size
        self.flip = flip

    def __iter__(self):
        self.stop_A1 = False
        self.stop_A2 = False
        self.stop_B1 = False
        self.stop_B2 = False
        self.data_loader_A1_iter = iter(self.data_loader_A1)
        self.data_loader_A2_iter = iter(self.data_loader_A2)
        self.data_loader_B1_iter = iter(self.data_loader_B1)
        self.data_loader_B2_iter = iter(self.data_loader_B2)
        self.iter = 0
        return self

    def __next__(self):
        A1, A1_paths = None, None
        A2, A2_paths = None, None
        B1, B1_paths = None, None
        B2, B2_paths = None, None

        try:
            A1, A1_paths = next(self.data_loader_A1_iter)
        except StopIteration:
            if A1 is None or A1_paths is None:
                self.stop_A1 = True
                self.data_loader_A1_iter = iter(self.data_loader_A1)
                A1, A1_paths = next(self.data_loader_A1_iter)

        try:
            A2, A2_paths = next(self.data_loader_A2_iter)
        except StopIteration:
            if A2 is None or A2_paths is None:
                self.stop_A2 = True
                self.data_loader_A2_iter = iter(self.data_loader_A2)
                A2, A2_paths = next(self.data_loader_A2_iter)

        try:
            B1, B1_paths = next(self.data_loader_B1_iter)
        except StopIteration:
            if B1 is None or B1_paths is None:
                self.stop_B1 = True
                self.data_loader_B1_iter = iter(self.data_loader_B1)
                B1, B1_paths = next(self.data_loader_B1_iter)

        try:
            B2, B2_paths = next(self.data_loader_B2_iter)
        except StopIteration:
            if B2 is None or B2_paths is None:
                self.stop_B2 = True
                self.data_loader_B2_iter = iter(self.data_loader_B2)
                B2, B2_paths = next(self.data_loader_B2_iter)


        if (self.stop_A1 and self.stop_A2 and self.stop_B1 and self.stop_B2) or self.iter > self.max_dataset_size:
            self.stop_A1 = False
            self.stop_A2 = False
            self.stop_B1 = False
            self.stop_B2 = False
            raise StopIteration()
        else:
            self.iter += 1
            if self.flip and random.random() < 0.5:
                idx = [i for i in range(A1.size(3) - 1, -1, -1)]
                idx = torch.LongTensor(idx)
                A1 = A1.index_select(3, idx)
                A2 = A2.index_select(3, idx)
                B1 = B1.index_select(3, idx)
                B2 = B2.index_select(3, idx)
            return {'A1': A1, 'A1_paths': A1_paths,
                    'A2': A2, 'A2_paths': A2_paths,
                    'B1': B1, 'B1_paths': B1_paths,
                    'B2': B2, 'B2_paths': B2_paths}

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

class FourAlignedDataLoader(BaseDataLoader):
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

        # Dataset A1, eg.. train/A1目录
        dataset_A1 = ImageFolder(root=opt.dataroot + '/' + opt.phase + '/A1', transform=transform, return_paths=True)
        data_loader_A1 = torch.utils.data.DataLoader(
            dataset_A1, batch_size=self.opt.batchSize, shuffle=not self.opt.serial_batches, num_workers=int(self.opt.nThreads))

        dataset_A2 = ImageFolder(root=opt.dataroot + '/' + opt.phase + '/A2', transform=transform, return_paths=True)
        data_loader_A2 = torch.utils.data.DataLoader(
            dataset_A2, batch_size=self.opt.batchSize, shuffle=not self.opt.serial_batches, num_workers=int(self.opt.nThreads))

        dataset_B1 = ImageFolder(root=opt.dataroot + '/' + opt.phase + '/B1', transform=transform, return_paths=True)
        data_loader_B1 = torch.utils.data.DataLoader(
            dataset_B1, batch_size=self.opt.batchSize, shuffle=not self.opt.serial_batches, num_workers=int(self.opt.nThreads))

        dataset_B2 = ImageFolder(root=opt.dataroot + '/' + opt.phase + '/B2', transform=transform, return_paths=True)
        data_loader_B2 = torch.utils.data.DataLoader(
            dataset_B2, batch_size=self.opt.batchSize, shuffle=not self.opt.serial_batches, num_workers=int(self.opt.nThreads))

        # 如何保证 A1、A2、B1、B2是一一一对应的呢，shuffle=not self.opt.serial_batches 这个参数 serial_batches 为True，代表有序，否则随机
        # shuffle 是洗牌，搅乱的意思
        #
        # 奇怪，如果A和B的数据数量不一致呢？也可以在opt里面修改load的数量大小的

        self.dataset_A1 = dataset_A1
        self.dataset_A2 = dataset_A2
        self.dataset_B1 = dataset_B1
        self.dataset_B2 = dataset_B2
        flip = opt.isTrain and not opt.no_flip
        self.four_paired_data = FourPairedData(data_loader_A1, data_loader_A2, data_loader_B1, data_loader_B2,self.opt.max_dataset_size, flip)

    def name(self):
        return 'FourAlignedDataLoader'

    def load_data(self):
        return self.four_paired_data

    def __len__(self):
        return min(max(len(self.dataset_A1), len(self.dataset_A2), len(self.dataset_B1), len(self.dataset_B2)), self.opt.max_dataset_size)
