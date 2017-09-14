#coding=utf8
import os
import sys
import subprocess

def mean(numbers):
    return float(sum(numbers)) / max(len(numbers), 1)

def ssim(src, target):
    # pip install pysssim
    args = ["pyssim", src, target]
    ssim_call = " ".join(args)
    p = subprocess.Popen(ssim_call, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    ssim_score = p.stdout.read()
    return ssim_score

def ssim_by_fold(path):
    ssim_list = []
    for root, dirs, files in os.walk(path):
        for f in files:
            if f.find("fake") != -1:
                fake_file = os.path.join(root, f)
                real_file = os.path.join(root, f.replace("fake", "real"))
                ssim_list.append(float(ssim(fake_file, real_file)))

    return mean(ssim_list)


if __name__=="__main__":
    # src = "/data/donghaoye/pytorch-CycleGAN-and-pix2pix/results/skeleton_pix2pix_abc_skeleton_ref_real_0818/test_100/images/person02_handwaving_d1_frame_222_298_skeleton_ske_frame_222_real_B.png"
    # target = "/data/donghaoye/pytorch-CycleGAN-and-pix2pix/results/skeleton_pix2pix_abc_skeleton_ref_real_0818/test_100/images/person02_handwaving_d1_frame_222_298_skeleton_ske_frame_222_fake_B.png"
    #src = "I:/person02_handwaving_d1_frame_222_298_skeleton_ske_frame_222_real_B.png"
    #target = "I:/person02_handwaving_d1_frame_222_298_skeleton_ske_frame_222_fake_B.png"
    #src_path = unicode(src, "utf8")
    #target_path = unicode(target, "utf8")
    #ssim_score = ssim(src_path, target_path)
    #print "ssim_score:" + ssim_score

    #path = "I:/20170826/test_latest/images"

    path = sys.argv[1]
    ssim = ssim_by_fold(path)

    print ssim