"""
# > Implementation of the classic paper by Zhou Wang et. al.: 
#     - Image quality assessment: from error visibility to structural similarity
#     - https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=1284395
# > Maintainer: https://github.com/xahidbuffon
#得到结构平均相似度和峰值信噪比的函数
"""
from __future__ import division#导入division(精确算法)以后，"/"执行的是精确值（小数）
import numpy as np
import math#导入math模块
from scipy.ndimage import gaussian_filter#从n维数据包中导入高斯滤波



def getSSIM(X, Y):
    """
       Computes the mean structural similarity between two images.计算两个图片间结构的平均相似度
    """
    assert (X.shape == Y.shape), "Image-patche provided have different dimensions"#判断表达式，在表达式条件为false的时候触发异常“有不同的尺寸时”将触发一个异常并显示错误消息"Image-patches provided have different dimensions"。
    nch = 1 if X.ndim==2 else X.shape[-1]#如果数组X的维度为2，nch=1，不然nch=数组X列（维度）的大小,这一行代码用于确定图像的通道数。如果图像的维度是2，则表示图像是灰度图像，通道数为1；否则，通过X.shape[-1]获取图像的最后一个维度，即通道数。
    mssim = []#mssim为数组对象,存储每个通道的结构相似度。
    for ch in range(nch):#ch遍历生成器内容，共nch次!!!!!!!!change!!!!!!!!!xrange(before)用于遍历每个通道。range(nch)将生成从0到nch-1的整数序列，作为循环的迭代器。
        Xc, Yc = X[...,ch].astype(np.float64), Y[...,ch].astype(np.float64)#将ch的所有dataframe字段转换为float64类型
        mssim.append(compute_ssim(Xc, Yc))#确保维度，将compute_ssim(Xc, Yc)（下方有函数的定义）看作一个对象，整体打包添加到mssim对象中
    return np.mean(mssim)#返回missm的平均值


def compute_ssim(X, Y):
    """
       Compute the structural similarity per single channel (given two images)。给两个图片计算每个单管道的结构相似性
    """
    # variables are initialized as suggested in the paper在文中表明了变量的初始化 
    K1 = 0.01
    K2 = 0.03
    sigma = 1.5
    win_size = 5   

    # means方式方法均值
    ux = gaussian_filter(X, sigma)#高斯滤波器，输入为X图像，sigma标量或标量序列，就是高斯函数里面的，这个值越大，滤波之后的图像越模糊，返回值是和输入形状一样的矩阵
    uy = gaussian_filter(Y, sigma)

    # variances and covariances变化幅度和协方差
    uxx = gaussian_filter(X * X, sigma)
    uyy = gaussian_filter(Y * Y, sigma)
    uxy = gaussian_filter(X * Y, sigma)

    # normalize by unbiased estimate of std dev 无偏估计
    N = win_size ** X.ndim
    unbiased_norm = N / (N - 1)  # eq. 4 of the paper
    vx  = (uxx - ux * ux) * unbiased_norm
    vy  = (uyy - uy * uy) * unbiased_norm
    vxy = (uxy - ux * uy) * unbiased_norm

    R = 255#即L,比特深度为8
    C1 = (K1 * R) ** 2
    C2 = (K2 * R) ** 2
    # compute SSIM (eq. 13 of the paper)
    sim = (2 * ux * uy + C1) * (2 * vxy + C2)
    D = (ux ** 2 + uy ** 2 + C1) * (vx + vy + C2)
    SSIM = sim/D 
    mssim = SSIM.mean()

    return mssim



def getPSNR(X, Y):
    #assume RGB image
    target_data = np.array(X, dtype=np.float64)
    ref_data = np.array(Y, dtype=np.float64)
    diff = ref_data - target_data
    diff = diff.flatten('C')
    rmse = math.sqrt(np.mean(diff ** 2.) )##求mse，此处用的均方根，后面就不用多乘个255
    if rmse == 0: return 100
    else: return 20*math.log10(255.0/rmse)

def getColourDistance(X, Y):
    #计算色差
    target_data = np.array(X, dtype=np.float64)
    ref_data = np.array(Y, dtype=np.float64)
    R_target, G_target, B_target = target_data[:, :, 0], target_data[:, :, 1], target_data[:, :, 2]
    R_ref, G_ref, B_ref = ref_data[:, :, 0], ref_data[:, :, 1], ref_data[:, :,2]
    rmean = np.mean((R_target + R_ref) / 2)
    R = np.mean(R_target - R_ref)
    G = np.mean(G_target - G_ref)
    B = np.mean(B_target - B_ref)
    return math.sqrt((2 + rmean / 256) * (R ** 2) + 4 * (G ** 2) + (2 + (255 - rmean) / 256) * (B ** 2))

def rgb_xyz(X):
    X_linear = np.where(X > 0.04045, ((X + 0.055) / 1.055) ** 2.4, X / 12.9)
    M = np.array([[0.4124, 0.3576, 0.1805],
                  [0.2126, 0.7152, 0.0722],
                  [0.0193, 0.1192, 0.9505]])
    xyz = np.dot(M, X_linear)

    return xyz.T

def xyz_lab(xyz):
    X_w, Y_w, Z_w = 0.95047, 1, 1.08883
    result = xyz / np.array([X_w, Y_w, Z_w])  # 实际上是X/Xn, Y/Yn, Z/Zn
    x_n = result[:, 0]
    y_n = result[:, 1]
    z_n = result[:, 2]
    # x_n, y_n, z_n = xyz / np.array([X_w, Y_w, Z_w])# 实际上是X/Xn, Y/Yn, Z/Zn
    mask = x_n > 0.008856
    fx = np.where(mask, x_n**(1.0 / 3.0), (903.3 * x_n + 16) / 116.0)

    mask = y_n > 0.008856
    fy = np.where(mask, y_n**(1.0 / 3.0), (903.3 * y_n + 16) / 116.0)

    mask = z_n > 0.008856
    fz = np.where(mask, z_n**(1.0 / 3.0), (903.3 * z_n + 16) / 116.0)

    L = (116.0 * fy) - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return np.array([L, a, b])


def getColourDifference(X, Y):
    # 计算LAB色差
    target_data = np.array(X, dtype=np.float64)
    ref_data = np.array(Y, dtype=np.float64)
    target_rgb = target_data / 255
    ref_rgb = ref_data / 255
    R_target, G_target, B_target = np.array(target_data[:, :, 0]).flatten(), np.array(target_data[:, :, 1]).flatten(), np.array(target_data[:, :, 2]).flatten()
    R_ref, G_ref, B_ref = np.array(ref_data[:, :, 0]).flatten(), np.array(ref_data[:, :, 1]).flatten(), np.array(ref_data[:, :,2]).flatten()
    target_rgb_new = np.array([R_target, G_target, B_target])

    ref_rgb_new = np.array([R_ref, G_ref, B_ref])

    target_xyz = rgb_xyz(target_rgb_new)
    ref_xyz = rgb_xyz(ref_rgb_new)

    target_lab = xyz_lab(target_xyz)
    ref_lab = xyz_lab(ref_xyz)

    L_target, a_target, b_target = target_lab[0, :], target_lab[1, :], target_lab[2, :]
    L_ref, a_ref, b_ref = ref_lab[0, :], ref_lab[1, :], ref_lab[2, :]

    L = np.mean(L_target - L_ref)
    a = np.mean(a_target - a_ref)
    b = np.mean(b_target - b_ref)

    return math.sqrt(L ** 2 + a ** 2 + b ** 2)

