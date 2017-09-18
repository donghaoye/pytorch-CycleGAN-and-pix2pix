import time
import os
from options.test_options import TestOptions
opt = TestOptions().parse()  # set CUDA_VISIBLE_DEVICES before import torch

from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from pdb import set_trace as st
from util import html

from compute_ssim import ssim
from compute_ssim import mean

opt.nThreads = 1   # test code only supports nThreads=1
opt.batchSize = 1  #test code only supports batchSize=1
opt.serial_batches = True # no shuffle

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)
visualizer = Visualizer(opt)
# create website
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

ssim_list = []

# test
for i, data in enumerate(dataset):
    if i >= opt.how_many:
        break
    model.set_input(data)
    model.test()
    visuals = model.get_current_visuals()

    #visualizer.display_current_results(model.get_current_visuals(), i)

    img_path = model.get_image_paths()
    if opt.model == 'pix2pix_abc':
        print('process image... {}, {}'.format( img_path["A1"], img_path["A2"]))
        visualizer.save_images(webpage, visuals, img_path["A1"])
        src = img_path["B"]
        target = src.replace("real", "fake")
        ssim_score = ssim(src, target)
        ssim_list.append(ssim_score)
        print("ssim : " + str(ssim_score))
    else:
        print('process image... %s' % img_path)
        visualizer.save_images(webpage, visuals, img_path)

now = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
ssim_result = mean(ssim_list)

log_path = os.path.join(web_dir, "ssim_result.txt")
log_file = open(log_path, "w")
log_file.write(now)
log_file.write(ssim_result)
webpage.save()
