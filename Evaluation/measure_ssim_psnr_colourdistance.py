"""
# > Script for measuring quantitative performances in terms of
#    - Structural Similarity Metric (SSIM) 
#    - Peak Signal to Noise Ratio (PSNR)
# > Maintainer: https://github.com/xahidbuffon
"""
## python libs
import numpy as np
import os
from PIL import Image
from glob import glob
from os.path import join
from ntpath import basename
## local libs
from imqual_utils import getSSIM, getPSNR, getColourDistance, getColourDifference#可以正常运行，不用管，要改搜unresolved reference

def SSIMs_PSNRs(gtr_dir, gen_dir, im_res=(400, 600)):#
    """
        - gtr_dir contain ground-truths
        - gen_dir contain generated images 
    """
    gtr_paths = sorted(glob(join(gtr_dir, "*.png")))#这行代码的作用是将gtr_dir目录下所有扩展名为.png的文件路径存储在gtr_paths列表中，并按照文件名进行排序。
    gen_paths = sorted(glob(join(gen_dir, "*.png")))
    ssims, psnrs, colour_distances, colour_differences = [], [], [], []
    for gtr_path, gen_path in zip(gtr_paths, gen_paths):
        gtr_f = basename(gtr_path).split('.')[0]#.split('_')[0]
        gen_f = basename(gen_path).split('.')[0]#.split('_')[0]
        if (gtr_f==gen_f):
            # assumes same filenames
            r_im = Image.open(gtr_path)#.resize(im_res)
            g_im = Image.open(gen_path)#.resize(im_res)

            ratio = np.mean(np.array(r_im))/np.mean(np.array(g_im))
            # 计算色差
            colour_distance = getColourDistance(np.array(r_im), np.array(g_im))
            colour_distances.append(colour_distance)
            # 重新计算色差
            colour_difference = getColourDifference(np.array(r_im), np.array(g_im))
            colour_differences.append(colour_difference)

            # get ssim on RGB channels
            ssim = getSSIM(np.array(r_im), np.array(g_im))
            ssims.append(ssim)
            r_im = r_im.convert("L"); g_im = g_im.convert("L")#可以改成计算彩色的三个通道，这里只用了一个通道,'L'代表转成灰度图像
            psnr = getPSNR(np.array(r_im), np.array(g_im))
            psnrs.append(psnr)
    return np.array(ssims), np.array(psnrs), np.array(colour_distances), np.array(colour_differences)


"""
Get datasets from
 - http://irvlab.cs.umn.edu/resources/euvp-dataset
 - http://irvlab.cs.umn.edu/resources/ufo-120-dataset
"""
gtr_dir = os.path.join("../data/Dataset_test", "LOLDataset/eval15/high/")
## generated im paths
gen_dir = os.path.join("../output_data/", "output_gen")
### compute SSIM and PSNR
SSIM_measures, PSNR_measures, Colour_distance_measures, Colour_different_measures = SSIMs_PSNRs(gtr_dir, gen_dir)


print ("SSIM on {0} samples".format(len(SSIM_measures)))
print ("Mean: {0} std: {1}".format(np.mean(SSIM_measures), np.std(SSIM_measures)))

print ("PSNR on {0} samples".format(len(PSNR_measures)))
print ("Mean: {0} std: {1}".format(np.mean(PSNR_measures), np.std(PSNR_measures)))

print ("Colour_Distance on {0} samples".format(len(Colour_distance_measures)))
print ("Mean: {0} std: {1}".format(np.mean(Colour_distance_measures), np.std(Colour_distance_measures)))

print ("Colour_Difference on {0} samples".format(len(Colour_different_measures)))
print ("Mean: {0} std: {1}".format(np.mean(Colour_different_measures), np.std(Colour_different_measures)))



