<img src='imgs/horse2zebra.gif' align="right" width=384>

<br><br><br>

# CycleGAN and pix2pix in PyTorch

This is our ongoing PyTorch implementation for both unpaired and paired image-to-image translation.

The code was written by [Jun-Yan Zhu](https://github.com/junyanz) and [Taesung Park](https://github.com/taesung89).

Check out the original [CycleGAN Torch](https://github.com/junyanz/CycleGAN) and [pix2pix Torch](https://github.com/phillipi/pix2pix) code if you would like to reproduce the exact same results as in the papers.


#### CycleGAN: [[Project]](https://junyanz.github.io/CycleGAN/) [[Paper]](https://arxiv.org/pdf/1703.10593.pdf) [[Torch]](https://github.com/junyanz/CycleGAN)
<img src="https://junyanz.github.io/CycleGAN/images/teaser_high_res.jpg" width="900"/>

#### Pix2pix:  [[Project]](https://phillipi.github.io/pix2pix/) [[Paper]](https://arxiv.org/pdf/1611.07004v1.pdf) [[Torch]](https://github.com/phillipi/pix2pix)

<img src="https://phillipi.github.io/pix2pix/images/teaser_v3.png" width="900px"/>

#### [[EdgesCats Demo]](https://affinelayer.com/pixsrv/)  [[pix2pix-tensorflow]](https://github.com/affinelayer/pix2pix-tensorflow)   
Written by [Christopher Hesse](https://twitter.com/christophrhesse)  

<img src='imgs/edges2cats.jpg' width="600px"/>

If you use this code for your research, please cite:

Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks  
[Jun-Yan Zhu](https://people.eecs.berkeley.edu/~junyanz/)\*,  [Taesung Park](https://taesung.me/)\*, [Phillip Isola](https://people.eecs.berkeley.edu/~isola/), [Alexei A. Efros](https://people.eecs.berkeley.edu/~efros/)  
In arxiv, 2017. (* equal contributions)  


Image-to-Image Translation with Conditional Adversarial Networks  
[Phillip Isola](https://people.eecs.berkeley.edu/~isola/), [Jun-Yan Zhu](https://people.eecs.berkeley.edu/~junyanz/), [Tinghui Zhou](https://people.eecs.berkeley.edu/~tinghuiz/), [Alexei A. Efros](https://people.eecs.berkeley.edu/~efros/)   
In CVPR 2017.



## Prerequisites
- Linux or OSX.
- Python 2 or Python 3.
- CPU or NVIDIA GPU + CUDA CuDNN.

## Getting Started
### Installation
- Install PyTorch and dependencies from http://pytorch.org/
- Install python libraries [visdom](https://github.com/facebookresearch/visdom) and [dominate](https://github.com/Knio/dominate).
```bash
pip install visdom
pip install dominate
```
- Clone this repo:
```bash
git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
cd pytorch-CycleGAN-and-pix2pix
```

### CycleGAN train/test
- Download a CycleGAN dataset (e.g. maps):
```bash
bash ./datasets/download_cyclegan_dataset.sh maps
```
- Train a model:
```bash
#!./scripts/train_cyclegan.sh
--align_data 2
python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
python train.py --dataroot ./datasets/stand2sit --name stand2sit_cyclegan --model cycle_gan --which_model_netG flownet
python train.py --dataroot /data/donghaoye/KTH/data_cycleGAN/skeleton --name skeleton_cyclegan --model cycle_gan
python train.py --dataroot /data/donghaoye/KTH/data5_cycleGAN/handwaving_walking --name handwaving_walking_cyclegan --model cycle_gan  --align_data 2



```
- To view training results and loss plots, run `python -m visdom.server` and click the URL http://localhost:8097. To see more intermediate results, check out `./checkpoints/maps_cyclegan/web/index.html`
- Test the model:
```bash
#!./scripts/test_cyclegan.sh
python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan --phase test
```
The test results will be saved to a html file here: `./results/maps_cyclegan/latest_test/index.html`.

### pix2pix train/test
- Download a pix2pix dataset (e.g.facades):
```bash
bash ./datasets/download_pix2pix_dataset.sh facades
```
- Train a model:
```bash
#!./scripts/train_pix2pix.sh


CUDA_VISIBLE_DEVICES=1, 2, 3 python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --which_model_netG unet_256 --which_direction BtoA --lambda_A 100 --align_data 1 --use_dropout --no_lsgan --display_id 0
CUDA_VISIBLE_DEVICES=1 python train.py --dataroot /home/disk2/donghaoye/KTH/data4/train_A_B --name skeleton_pix2pix --model pix2pix --which_model_netG unet_256 --which_direction BtoA --niter_decay 10 --niter 10 --align_data --use_dropout --no_lsgan
CUDA_VISIBLE_DEVICES=0 python train.py --dataroot /home/disk2/donghaoye/KTH/data4/train_A_B --name skeleton_pix2pix --model pix2pix --which_model_netG flownet --which_direction BtoA --niter_decay 10 --niter 10 --align_data --use_dropout --no_lsgan
CUDA_VISIBLE_DEVICES=1 python train.py --dataroot /home/disk2/donghaoye/KTH/data4/train_A_B --name skeleton_pix2pix --model pix2pix --which_model_netG sia_unet --which_direction BtoA --niter_decay 10 --niter 10 --align_data --use_dropout --no_lsgan


CUDA_VISIBLE_DEVICES=1 python train.py --dataroot /home/disk2/donghaoye/KTH/data4/train_A_B --name skeleton_pix2pix --model pix2pix --which_model_netG unet_256 --which_direction BtoA --niter_decay 10 --niter 10 --align_data 1 --use_dropout --no_lsgan --serial_batches

CUDA_VISIBLE_DEVICES=3 python train.py --dataroot /home/disk2/donghaoye/KTH/data6/train_A_B_C/train --name skeleton_pix2pix_abc --model pix2pix_abc --which_model_netG sia_unet --niter_decay 20 --niter 20  --serial_batches --use_dropout --no_lsgan --align_data 3

加上 --serial_batches 就代表是True了(因为是action='store_true')，保证了获取的目录有序，但是训练的时候，是无序？？



# trianing in 51122
CUDA_VISIBLE_DEVICES=2,3 python train.py --dataroot /data/donghaoye/KTH/data8_skeleton_ref_real/train_ske_ref_img/handwaving --name skeleton_pix2pix_abc_skeleton_ref_real_0919 --model pix2pix_abc --which_model_netG sia_unet --niter_decay 100 --niter 100  --serial_batches --use_dropout --no_lsgan --align_data 3 --display_freq 1 --print_freq 1 --no_flip --gpu_ids 0,1
CUDA_VISIBLE_DEVICES=1 python train.py --dataroot /data/donghaoye/KTH/data8_skeleton_ref_real/train_ske_ref_img/handwaving --name skeleton_pix2pix_abc_skeleton_ref_real_0818 --model pix2pix_abc --which_model_netG sia_unet --niter_decay 100 --niter 100  --serial_batches --use_dropout --no_lsgan --align_data 3 --display_freq 1 --print_freq 1 --no_flip

CUDA_VISIBLE_DEVICES=2,3 python train.py --dataroot /home/disk2/donghaoye/KTH/data8_skeleton_ref_real/train_ske_ref_img/handwaving --name skeleton_pix2pix_abc_skeleton_ref_real_0919 --model pix2pix_abc --which_model_netG sia_unet --niter_decay 50 --niter 50  --serial_batches --use_dropout --no_lsgan --align_data 3 --display_freq 1 --print_freq 1 --gpu_ids 0,1


CUDA_VISIBLE_DEVICES=1 python train.py --dataroot /data/donghaoye/KTH/data8_skeleton_ref_real/train_ske_ref_img/handwaving --name skeleton_pix2pix_abc_skeleton_ref_real_0826 --model pix2pix_abc --which_model_netG sia_unet --niter_decay 20 --niter 20  --serial_batches --use_dropout --no_lsgan --align_data 3 --display_freq 1 --print_freq 1 --no_flip　--input_nc　6
CUDA_VISIBLE_DEVICES=1 python train.py --dataroot /data/donghaoye/KTH/data8_skeleton_ref_real/train_ske_ref_img/handwaving --name skeleton_pix2pix_abc_skeleton_ref_real_0826 --model pix2pix_abc --which_model_netG stack_unet --niter_decay 20 --niter 20  --serial_batches --use_dropout --no_lsgan --align_data 3 --display_freq 1 --print_freq 1 --no_flip --input_nc 6

CUDA_VISIBLE_DEVICES=1 python train.py --dataroot /home/disk2/donghaoye/KTH/data8_skeleton_ref_real/train_ske_ref_img/handwaving --name skeleton_pix2pix_abc_skeleton_ref_real_0826 --model pix2pix_abc --which_model_netG sia_unet --niter_decay 20 --niter 20  --serial_batches --use_dropout --no_lsgan --align_data 3 --display_freq 1 --print_freq 1 --no_flip --input_nc 3

CUDA_VISIBLE_DEVICES=1 python train.py --dataroot /home/disk2/donghaoye/KTH/data9_skeleton_ref_real/train_ske_ref_img/all/ --name skeleton_pix2pix_abc_skeleton_ref_real_all_novgg_0826 --model pix2pix_abc --which_model_netG sia_unet --niter_decay 20 --niter 20  --serial_batches --use_dropout --no_lsgan --align_data 3 --display_freq 200 --print_freq 200 --no_flip


# siamese stack 级联
CUDA_VISIBLE_DEVICES=1 python train.py --dataroot /data/donghaoye/KTH/data9_skeleton_ref_real/train/all/ --name skeleton_GAN_L1_pose_vgg_0925 --model pix2pix_abc --which_model_netG sia_stack_unet --niter_decay 20 --niter 20  --serial_batches --use_dropout --no_lsgan --align_data 3 --display_freq 20 --print_freq 20 --no_flip --input_nc 3 --serial_batches --use_vgg --use_pose --batchSize 10 --val_path xxxx

CUDA_VISIBLE_DEVICES=3 python train.py --dataroot /home/disk2/donghaoye/KTH/data9_skeleton_ref_real/train_ske_ref_img/all --name skeleton_pix2pix_abc_skeleton_ref_real_all_sia_stack0904 --model pix2pix_abc --which_model_netG sia_stack_unet --niter_decay 20 --niter 20 --use_dropout --no_lsgan --align_data 3 --display_freq 20 --print_freq 20 --no_flip --input_nc 3 --serial_batches


skeleton数据集
CUDA_VISIBLE_DEVICES=2 python train.py --dataroot /data/donghaoye/KTH/data9_skeleton_ref_real/train/all/ --name skeleton_pix2pix_abc_skeleton_ref_real_all_0914 --model pix2pix_abc --which_model_netG sia_unet --niter_decay 200 --niter 200  --serial_batches --use_dropout --no_lsgan --align_data 3 --display_freq 200 --print_freq 200 --no_flip


mpii数据集
CUDA_VISIBLE_DEVICES=0 python train.py --dataroot /data/donghaoye/datasets/mpii_all/mpii_pair/train --name skeleton_pix2pix_abc_skeleton_ref_real_all_sia_stack0931 --model pix2pix_abc --which_model_netG sia_stack_unet --niter_decay 20 --niter 20  --serial_batches --use_dropout --no_lsgan --align_data 3 --display_freq 1 --print_freq 1 --no_flip --input_nc 3

fashion数据集
CUDA_VISIBLE_DEVICES=3 python train.py --dataroot /data/donghaoye/datasets/In-shop_AB/train --name skeleton_pix2pix_abc_skeleton_ref_real_all_sia_stack_fashion_0907 --model pix2pix_abc --which_model_netG sia_stack_unet --niter_decay 20 --niter 20  --serial_batches --use_dropout --no_lsgan --align_data 3 --display_freq 1 --print_freq 1 --no_flip --input_nc 3
11111号机
CUDA_VISIBLE_DEVICES=0,1,2 python train.py --dataroot /home/disk2/donghaoye/In-shop_AB/train --name skeleton_pix2pix_abc_skeleton_ref_real_all_sia_stack_fashion_0920_epoch40 --model pix2pix_abc --which_model_netG sia_stack_unet --niter_decay 20 --niter 20  --serial_batches --use_dropout --no_lsgan --align_data 3 --display_freq 50 --print_freq 50 --no_flip --input_nc 3 --gpu_ids 0,1,2 --batchSize 10 --val_path /home/disk2/donghaoye/In-shop_AB/val


youtube数据集
CUDA_VISIBLE_DEVICES=3 python train.py --dataroot /data/donghaoye/datasets/youtube_pose/train_test/train --name skeleton_pix2pix_abc_skeleton_ref_real_all_sia_stack_youtube_0920 --model pix2pix_abc --which_model_netG sia_stack_unet --niter_decay 20 --niter 20  --serial_batches --use_dropout --no_lsgan --align_data 3 --display_freq 100 --print_freq 100 --no_flip --input_nc 3 --batchSize 10 --val_path xxxx


天河二号
 opt.display_id > 0 使用visdom
yhrun -n 4 -w gn[06-09] -p gpu CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --dataroot /HOME/sysu_issjyin_1/BIGDATA/donghaoye/KTH/data9_skeleton_ref_real/train/all --name skeleton_pix2pix_abc_skeleton_ref_real_all_0919 --model pix2pix_abc --which_model_netG sia_unet --niter_decay 100 --niter 100  --serial_batches --use_dropout --no_lsgan --align_data 3 --display_freq 100 --print_freq 100 --no_flip --gpu_ids 0,1,2,3 --batchSize 10 --val_path /HOME/sysu_issjyin_1/BIGDATA/donghaoye/KTH/data9_skeleton_ref_real/val

```
- To view training results and loss plots, run `python -m visdom.server` and click the URL http://localhost:8097. To see more intermediate results, check out  `./checkpoints/facades_pix2pix/web/index.html`
- Test the model (`bash ./scripts/test_pix2pix.sh`):
```bash
#!./scripts/test_pix2pix.sh
python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --which_model_netG unet_256 --which_direction BtoA --align_data
CUDA_VISIBLE_DEVICES=1 python test.py --dataroot /home/disk2/donghaoye/KTH/data4/test_A_B --name skeleton_pix2pix --model pix2pix --which_model_netG unet_256 --which_direction BtoA  --align_data

CUDA_VISIBLE_DEVICES=1 python test.py --dataroot /home/disk2/donghaoye/KTH/data6/test_A_B_C/test --name skeleton_pix2pix_abc_siamese_madebystack_20170815 --model pix2pix_abc --which_model_netG sia_unet --serial_batches --align_data 3 --which_epoch latest


CUDA_VISIBLE_DEVICES=1 python test.py --dataroot /home/disk2/donghaoye/KTH/data8_skeleton_ref_real/test_ske_ref_img/handwaving --name skeleton_pix2pix_abc_skeleton_ref_real_0824 --model pix2pix_abc --which_model_netG sia_unet --serial_batches --align_data 3 --which_epoch 10


CUDA_VISIBLE_DEVICES=1 python test.py --dataroot /data/donghaoye/KTH/data8_skeleton_ref_real/test_ske_ref_img/handwaving --name skeleton_pix2pix_abc_skeleton_ref_real_0818 --model pix2pix_abc --which_model_netG sia_unet --serial_batches --align_data 3 --which_epoch latest

CUDA_VISIBLE_DEVICES=1 python test.py --dataroot /home/disk2/donghaoye/KTH/data8_skeleton_ref_real/test_ske_ref_img/handwaving --name skeleton_pix2pix_abc_skeleton_ref_real --model pix2pix_abc --which_model_netG sia_unet --serial_batches --align_data 3 --which_epoch 10



```
The test results will be saved to a html file here: `./results/facades_pix2pix/latest_val/index.html`.

More example scripts can be found at `scripts` directory.

## Training/test Details
- See `options/train_options.py` and `options/base_options.py` for training flags; see `options/test_options.py` and `options/base_options.py` for test flags.
- CPU/GPU (default `--gpu_ids 0`): Set `--gpu_ids -1` to use CPU mode; set `--gpu_ids 0,1,2` for multi-GPU mode. You need a large batch size (e.g. `--batchSize 32`) to benefit from multiple gpus.  
- During training, the current results can be viewed using two methods. First, if you set `--display_id` > 0, the results and loss plot will be shown on a local graphics web server launched by [visdom](https://github.com/facebookresearch/visdom). To do this, you should have visdom installed and a server running by the command `python -m visdom.server`. The default server URL is `http://localhost:8097`. `display_id` corresponds to the window ID that is displayed on the `visdom` server. The `visdom` display functionality is turned on by default. To avoid the extra overhead of communicating with `visdom` set `--display_id 0`. Second, the intermediate results are saved to `[opt.checkpoints_dir]/[opt.name]/web/` as an HTML file. To avoid this, set `--no_html`.

### CycleGAN Datasets
Download the CycleGAN datasets using the following script:
```bash
bash ./datasets/download_cyclegan_dataset.sh dataset_name
```
- `facades`: 400 images from the [CMP Facades dataset](http://cmp.felk.cvut.cz/~tylecr1/facade/).
- `cityscapes`: 2975 images from the [Cityscapes training set](https://www.cityscapes-dataset.com/).
- `maps`: 1096 training images scraped from Google Maps.
- `horse2zebra`: 939 horse images and 1177 zebra images downloaded from [ImageNet](http://www.image-net.org/) using keywords `wild horse` and `zebra`
- `apple2orange`: 996 apple images and 1020 orange images downloaded from [ImageNet](http://www.image-net.org/) using keywords `apple` and `navel orange`.
- `summer2winter_yosemite`: 1273 summer Yosemite images and 854 winter Yosemite images were downloaded using Flickr API. See more details in our paper.
- `monet2photo`, `vangogh2photo`, `ukiyoe2photo`, `cezanne2photo`: The art images were downloaded from [Wikiart](https://www.wikiart.org/). The real photos are downloaded from Flickr using the combination of the tags *landscape* and *landscapephotography*. The training set size of each class is Monet:1074, Cezanne:584, Van Gogh:401, Ukiyo-e:1433, Photographs:6853.
- `iphone2dslr_flower`: both classes of images were downlaoded from Flickr. The training set size of each class is iPhone:1813, DSLR:3316. See more details in our paper.

To train a model on your own datasets, you need to create a data folder with two subdirectories `trainA` and `trainB` that contain images from domain A and B. You can test your model on your training set by setting ``phase='train'`` in  `test.lua`. You can also create subdirectories `testA` and `testB` if you have test data.

You should **not** expect our method to work on just any random combination of input and output datasets (e.g. `cats<->keyboards`). From our experiments, we find it works better if two datasets share similar visual content. For example, `landscape painting<->landscape photographs` works much better than `portrait painting <-> landscape photographs`. `zebras<->horses` achieves compelling results while `cats<->dogs` completely fails.

### pix2pix datasets
Download the pix2pix datasets using the following script:
```bash
bash ./datasets/download_pix2pix_dataset.sh dataset_name
```
- `facades`: 400 images from [CMP Facades dataset](http://cmp.felk.cvut.cz/~tylecr1/facade/).
- `cityscapes`: 2975 images from the [Cityscapes training set](https://www.cityscapes-dataset.com/).
- `maps`: 1096 training images scraped from Google Maps
- `edges2shoes`: 50k training images from [UT Zappos50K dataset](http://vision.cs.utexas.edu/projects/finegrained/utzap50k/). Edges are computed by [HED](https://github.com/s9xie/hed) edge detector + post-processing.
- `edges2handbags`: 137K Amazon Handbag images from [iGAN project](https://github.com/junyanz/iGAN). Edges are computed by [HED](https://github.com/s9xie/hed) edge detector + post-processing.

We provide a python script to generate pix2pix training data in the form of pairs of images {A,B}, where A and B are two different depictions of the same underlying scene. For example, these might be pairs {label map, photo} or {bw image, color image}. Then we can learn to translate A to B or B to A:

Create folder `/path/to/data` with subfolders `A` and `B`. `A` and `B` should each have their own subfolders `train`, `val`, `test`, etc. In `/path/to/data/A/train`, put training images in style A. In `/path/to/data/B/train`, put the corresponding images in style B. Repeat same for other data splits (`val`, `test`, etc).

Corresponding images in a pair {A,B} must be the same size and have the same filename, e.g., `/path/to/data/A/train/1.jpg` is considered to correspond to `/path/to/data/B/train/1.jpg`.

Once the data is formatted this way, call:
```bash
python datasets/combine_A_and_B.py --fold_A /path/to/data/A --fold_B /path/to/data/B --fold_AB /path/to/data
```

This will combine each pair of images (A,B) into a single image file, ready for training.

## TODO
- add reflection and other padding layers.
- add more preprocessing options.

## Related Projects:
[CycleGAN](https://github.com/junyanz/CycleGAN): Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks  
[pix2pix](https://github.com/phillipi/pix2pix): Image-to-image translation with conditional adversarial nets  
[iGAN](https://github.com/junyanz/iGAN): Interactive Image Generation via Generative Adversarial Networks

## Cat Paper Collection
If you love cats, and love reading cool graphics, vision, and learning papers, please check out the Cat Paper Collection:  
[[Github]](https://github.com/junyanz/CatPapers) [[Webpage]](http://people.eecs.berkeley.edu/~junyanz/cat/cat_papers.html)

## Acknowledgments
Code is inspired by [pytorch-DCGAN](https://github.com/pytorch/examples/tree/master/dcgan).

GOOD

2017-09-25
sia_stack 的 GAN + L1 + pose + VGG
CUDA_VISIBLE_DEVICES=1 python train.py --dataroot /data/donghaoye/KTH/data9_skeleton_ref_real/train/all/ --name KTH_sia_stack_GAN_L1_pose_vgg_0925 \
--model pix2pix_abc --which_model_netG sia_stack_unet --niter_decay 20 --niter 20  --serial_batches --use_dropout --no_lsgan --align_data 3 \
 --display_freq 20 --print_freq 20 --no_flip --input_nc 3 --serial_batches --use_vgg --use_pose --batchSize 5 --val_path xxxx


sia的GAN + L1
CUDA_VISIBLE_DEVICES=1 python train.py --dataroot /home/disk2/donghaoye/KTH/data9_skeleton_ref_real/train_ske_ref_img/all \
--name KTH_sia_GAN_L1_0925 --model pix2pix_abc --which_model_netG sia_unet --niter_decay 20 --niter 20  --serial_batches \
--use_dropout --no_lsgan --align_data 3 --display_freq 20 --print_freq 20 --no_flip --input_nc 3 --serial_batches --batchSize 5 --val_path xxxx




2017-09-28

cycleGAN_KTH_GAN_L1_0928
CUDA_VISIBLE_DEVICES=0,1 python train.py --dataroot /home/disk2/donghaoye/KTH/data10_handwaving_walking \
 --name cycleGAN_KTH_GAN_L1_0928 --model cycle_gan_aabb --no_flip --serial_batches --use_dropout --no_lsgan \
 --which_model_netG sia_unet --niter_decay 20 --niter 20 --align_data 4 \
 --display_freq 20 --print_freq 20 --input_nc 3  --batchSize 5 --gpu_ids 0,1 --identity 1

========================================================================================
cycleGAN_KTH_GAN_L1_bilinear_0928  将原始的逆卷积变为双线性插值的上采样
CUDA_VISIBLE_DEVICES=2,3 python train.py --dataroot /home/disk2/donghaoye/KTH/data10_handwaving_walking \
 --name cycleGAN_KTH_GAN_L1_Bi_0928 --model cycle_gan_aabb --no_flip --serial_batches --use_dropout --no_lsgan \
 --which_model_netG sia_unet_bilinear --niter_decay 20 --niter 20 --align_data 4 \
 --display_freq 20 --print_freq 20 --input_nc 3  --batchSize 1 --gpu_ids 0,1 --identity 1



 2017-09-29
 old_cycleGAN_KTH_GAN_L1_0929
CUDA_VISIBLE_DEVICES=1 python train.py --dataroot /home/disk2/donghaoye/KTH/data10_handwaving_walking \
 --name old_cycleGAN_KTH_GAN_L1_0928 --model cycle_gan_aabb --no_flip --serial_batches --use_dropout --no_lsgan \
 --which_model_netG sia_unet --niter_decay 20 --niter 20 --align_data 4 \
 --display_freq 20 --print_freq 20 --input_nc 3  --batchSize 5 --gpu_ids 0 --identity 0


==================================================================================
2017-10-04
CUDA_VISIBLE_DEVICES=2 python test.py --dataroot /home/disk2/donghaoye/KTH/data9_skeleton_ref_real/test_ske_ref_img/all \
--name KTH_sia_GAN_L1_0925 \
--model pix2pix_abc --which_model_netG sia_unet --serial_batches --align_data 3 --which_epoch latest --val_path xxxx

CUDA_VISIBLE_DEVICES=2,3 python test.py --dataroot /data/donghaoye/KTH/data9_skeleton_ref_real/test/all/  \
--name KTH_sia_stack_GAN_L1_pose_vgg_0925 \
--model pix2pix_abc --which_model_netG sia_unet --serial_batches --align_data 3 --which_epoch latest --val_path xxxx --gpu_ids 0,1
==================================================================================



2017-10-11
cycleGAN_KTH_GAN_L1_1011
self.loss_G = self.loss_G_A + self.loss_cycle_A + self.loss_idt_A 只是保留了一边的G_A loss
------------------------------------------
CUDA_VISIBLE_DEVICES=2 python train.py --dataroot /data/donghaoye/KTH/data10_handwaving_walking \
 --name cycleGAN_KTH_GAN_L1_1011 --model cycle_gan_aabb --no_flip --serial_batches --use_dropout --no_lsgan \
 --which_model_netG sia_unet --niter_decay 50 --niter 50 --align_data 4 \
 --display_freq 20 --print_freq 20 --input_nc 3  --batchSize 5 --gpu_ids 0

cycleGAN_KTH_GAN_L1_All_1011
self.loss_G = self.loss_G_A + self.loss_cycle_A + self.loss_idt_A  利用原始的cycleGAN
------------------------------------------
CUDA_VISIBLE_DEVICES=3 python train.py --dataroot /data/donghaoye/KTH/data10_handwaving_walking \
 --name oldCycleGAN_KTH_GAN_L1_1011 --model cycle_gan_aabb --no_flip --serial_batches --use_dropout --no_lsgan \
 --which_model_netG sia_unet --niter_decay 50 --niter 50 --align_data 4 \
 --display_freq 20 --print_freq 20 --input_nc 3  --batchSize 5 --gpu_ids 0


self.loss_G_A + self.loss_idt_A
cycleGAN_KTH_GAN_L1_only_1011
self.loss_G = self.loss_G_A + self.loss_cycle_A + self.loss_idt_A  利用原始的cycleGAN
------------------------------------------
CUDA_VISIBLE_DEVICES=1 python train.py --dataroot /data/donghaoye/KTH/data10_handwaving_walking \
 --name cycleGAN_KTH_GAN_L1_only_1011 --model cycle_gan_aabb --no_flip --serial_batches --use_dropout --no_lsgan \
 --which_model_netG sia_unet --niter_decay 100 --niter 100 --align_data 4 \
 --display_freq 20 --print_freq 20 --input_nc 3  --batchSize 5 --gpu_ids 0



 2017-10-11

self.loss_G = self.loss_G_A + self.loss_cycle_A + self.loss_idt_A 只是保留了一边的G_A loss
------------------------------------------
CUDA_VISIBLE_DEVICES=3 python train.py --dataroot /home/disk2/donghaoye/KTH/data10_handwaving_walking \
 --name cycleGAN_KTH_GAN_L1_1011 --model cycle_gan_aabb --no_flip --serial_batches --use_dropout --no_lsgan \
 --which_model_netG sia_unet --niter_decay 50 --niter 50 --align_data 4 \
 --display_freq 100 --print_freq 100 --input_nc 3  --batchSize 1 --gpu_ids 0



>>>>>>>>>>>>>>>>>>>>>>>美丽的分割线>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


  2017-10-17
自定义predict_z
这里的/data/donghaoye/KTH/data11_predict_z/trainC 是从data9里面直接获取
  CUDA_VISIBLE_DEVICES=1 python train.py --dataroot /data/donghaoye/KTH/data11_predict_z/train/trainC --name KTH_predict_z_L1_1017 \
--model predict_z --which_model_netG predict_z --niter_decay 20 --niter 20  --serial_batches --use_dropout --no_lsgan --align_data 5 \
 --display_freq 100 --print_freq 100 --no_flip --input_nc 3 --serial_batches --batchSize 1 --gpu_ids 0

CUDA_VISIBLE_DEVICES=2 python train.py --dataroot /data/donghaoye/KTH/data11_predict_z/train/trainC --name KTH_predict_z_L1_vgg_1017 \
--model predict_z --which_model_netG predict_z --niter_decay 20 --niter 20  --serial_batches --use_dropout --no_lsgan --align_data 5 \
 --display_freq 100 --print_freq 100 --no_flip --input_nc 3 --serial_batches --batchSize 1 --gpu_ids 0

 CUDA_VISIBLE_DEVICES=0 python train.py --dataroot /data/donghaoye/KTH/data11_predict_z/train/trainC --name KTH_predict_z_GAN_L1_vgg_1017 \
--model predict_z --which_model_netG predict_z --niter_decay 20 --niter 20  --serial_batches --use_dropout --no_lsgan --align_data 5 \
 --display_freq 100 --print_freq 100 --no_flip --input_nc 3 --serial_batches --batchSize 1 --gpu_ids 0 --use_gan

增加获取z0 和 z1
CUDA_VISIBLE_DEVICES=3 python train.py --dataroot /data/donghaoye/KTH/data9_skeleton_ref_real/train/all/ --name KTH_en_de_GAN_L1_pose_vgg_1017 \
--model pix2pix_abc --which_model_netG sia_en_de --niter_decay 20 --niter 20  --serial_batches --use_dropout --no_lsgan --align_data 3 \
 --display_freq 100 --print_freq 100 --no_flip --input_nc 3 --serial_batches --use_vgg --use_pose --batchSize 1 --gpu_ids 0

#原始版本
CUDA_VISIBLE_DEVICES=2,3 python train.py --dataroot /data/donghaoye/KTH/data8_skeleton_ref_real/train_ske_ref_img/handwaving \
--name skeleton_pix2pix_abc_skeleton_ref_real_0919 --model pix2pix_abc --which_model_netG sia_unet --niter_decay 100 --niter 100 \
--serial_batches --use_dropout --no_lsgan --align_data 3 --display_freq 1 --print_freq 1 --no_flip --gpu_ids 0,1



2017-10-24
#testing!!!
CUDA_VISIBLE_DEVICES=2 python test.py --dataroot /data/donghaoye/KTH/data11_predict_z/test/testC  \
--name KTH_predict_z_L1_1017 --model predict_z --which_model_netG predict_z \
--align_data 5 --input_nc 3 --serial_batches --batchSize 1 --gpu_ids 0

CUDA_VISIBLE_DEVICES=2 python test.py --dataroot /data/donghaoye/KTH/data11_predict_z/test/testC  \
--name KTH_predict_z_L1_vgg_1017 --model predict_z --which_model_netG predict_z \
--align_data 5 --input_nc 3 --serial_batches --batchSize 1 --gpu_ids 0

CUDA_VISIBLE_DEVICES=2 python test.py --dataroot /data/donghaoye/KTH/data11_predict_z/test/testC  \
--name KTH_predict_z_GAN_L1_vgg_1017 --model predict_z --which_model_netG predict_z \
--align_data 5 --input_nc 3 --serial_batches --batchSize 1 --gpu_ids 0

CUDA_VISIBLE_DEVICES=2 python test.py --dataroot /data/donghaoye/KTH/data9_skeleton_ref_real/test/all/  \
--name KTH_en_de_GAN_L1_pose_vgg_1017 --model pix2pix_abc --which_model_netG sia_en_de \
--align_data 3 --input_nc 3 --serial_batches --batchSize 1 --gpu_ids 0

#原始版本
CUDA_VISIBLE_DEVICES=2 python test.py --dataroot /data/donghaoye/KTH/data9_skeleton_ref_real/test/all/  \
--name skeleton_pix2pix_abc_skeleton_ref_real_0919 --model pix2pix_abc --which_model_netG sia_unet \
--align_data 3 --input_nc 3 --serial_batches --batchSize 1 --gpu_ids 0


CUDA_VISIBLE_DEVICES=2 python test.py --dataroot /data/donghaoye/KTH/selected_pointList/test_data/hand  \
--name skeleton_pix2pix_abc_skeleton_ref_real_0919 --model pix2pix_abc --which_model_netG sia_unet \
--align_data 3 --input_nc 3 --serial_batches --batchSize 1 --gpu_ids 0 \
--results_dir /data/donghaoye/KTH/selected_pointList/test_data/hand/results/

CUDA_VISIBLE_DEVICES=3 python test.py --dataroot /data/donghaoye/KTH/selected_pointList/test_data/hand2run  \
--name skeleton_pix2pix_abc_skeleton_ref_real_0919 --model pix2pix_abc --which_model_netG sia_unet \
--align_data 3 --input_nc 3 --serial_batches --batchSize 1 --gpu_ids 0 \
--results_dir /data/donghaoye/KTH/selected_pointList/test_data/hand2run/results/