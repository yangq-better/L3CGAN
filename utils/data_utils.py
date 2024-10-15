from __future__ import division
from __future__ import absolute_import
import os
import random
import fnmatch
import numpy as np
from PIL import Image
def deprocess(x, np_uint8=True):
    # [0,1] -> [0, 255]
    x = x*255#(x+1.0)*127.5
    return np.uint8(x) if np_uint8 else x
def preprocess(x):
    # [0,255] -> [0, 1]
    return x/255

def random_crop(a_img, b_img, patch_size=(256,256)):
    h = random.randint(0, a_img.shape[0]-patch_size[0])
    w = random.randint(0, a_img.shape[1]-patch_size[1])
    a_crop_img = a_img[h:h+patch_size[0], w:w+patch_size[1], :]
    b_crop_img = b_img[h:h+patch_size[0], w:w+patch_size[1], :]
    return a_crop_img, b_crop_img#随机裁剪

def augment(a_img, b_img):
    """
       Augment images - a is distorted
    """
    # randomly interpolate随机的更改
    a = random.random()
    a_noise = 255*np.random.rand(a_img.shape[0],a_img.shape[1],a_img.shape[2])#0-1之间均匀分布随机数
    # a_img = a_img*(1-a) + a_noise*a
    a_img = a_img + a_noise*a
    # flip image left right左右翻转图像
    if (random.random() < 0.25):
        a_img = np.fliplr(a_img)
        b_img = np.fliplr(b_img)
    # flip image up down上下翻转图像
    if (random.random() < 0.25):
        a_img = np.flipud(a_img)
        b_img = np.flipud(b_img)
    return a_img, b_img#翻转创造更多图像

def gain_ratio(a_img, b_img):
    avg_a = np.mean(a_img)
    avg_b = np.mean(b_img)
    ratio = int(avg_b/avg_a)
    a_img = a_img*ratio# 进行照度增强之后需要去掉？？
    # max_a_img = np.max(np.max(a_img, axis=-1))
    a_img = np.minimum(a_img, 255)#a_img*255/max_a_img
    return a_img, b_img
def getPaths(data_dir):
    exts = ['*.png','*.PNG','*.jpg','*.JPG', '*.JPEG']
    image_paths = []
    for pattern in exts:
        for d, s, fList in os.walk(data_dir):
            for filename in fList:
                if (fnmatch.fnmatch(filename, pattern)):
                    fname_ = os.path.join(d,filename)
                    image_paths.append(fname_)
    image_paths.sort()
    return np.asarray(image_paths)
def read(path):
    im = Image.open(path)#.resize(img_res)
    #print(im)
    if im.mode=='L':
        copy = np.zeros((im.shape[1], im.shape[0], 3))
        copy[:, :, 0] = im
        copy[:, :, 1] = im
        copy[:, :, 2] = im
        im = copy
    return np.array(im).astype(np.float32)
def read_pair(pathA, pathB):
    img_A = read(pathA)
    img_B = read(pathB)
    return img_A, img_B
def get_local_test_data(data_dir):
    assert os.path.exists(data_dir), "local image path doesnt exist"
    imgs = []
    for p in getPaths(data_dir):
        img = read(p)
        imgs.append(img)
    imgs = preprocess(np.array(imgs))
    return imgs
class DataLoader():
    def __init__(self, data_dir, dataset_name, test_only=False):
        #self.img_res = img_res
        self.DATA = dataset_name
        self.data_dir = data_dir
        if not test_only:
            self.trainA_paths = getPaths(os.path.join(self.data_dir, "low")) # low
            self.trainB_paths = getPaths(os.path.join(self.data_dir, "high")) # high
            if (len(self.trainA_paths)<len(self.trainB_paths)):
                self.trainB_paths = self.trainB_paths[:len(self.trainA_paths)]
            elif (len(self.trainA_paths)>len(self.trainB_paths)):
                self.trainA_paths = self.trainA_paths[:len(self.trainB_paths)]
            else: pass
            self.val_paths = getPaths(os.path.join(self.data_dir, "high"))
            self.num_train, self.num_val = len(self.trainA_paths), len(self.val_paths)
            print ("{0} training pairs\n".format(self.num_train))
        else:
            self.test_paths    = getPaths(os.path.join(self.data_dir, "test"))
            print ("{0} test images\n".format(len(self.test_paths)))

    def get_test_data(self, batch_size=1):
        idx = np.random.choice(np.arange(len(self.test_paths)), batch_size, replace=False)
        paths = self.test_paths[idx]
        imgs = []
        for p in paths:
            img = read(p)
            imgs.append(img)
        imgs = preprocess(np.array(imgs))
        return imgs

    def load_val_data(self, batch_size=1, patch_size=(256,256)):
        idx = np.random.choice(np.arange(self.num_val), batch_size, replace=False)
        pathsA = self.trainA_paths[idx]
        pathsB = self.trainB_paths[idx]
        imgs_A, imgs_B = [], []
        for idx in range(len(pathsB)):
            img_A, img_B = read_pair(pathsA[idx], pathsB[idx])
            img_A, img_B = random_crop(img_A, img_B, patch_size)
            img_A, img_B = gain_ratio(img_A, img_B)
            imgs_A.append(img_A)
            imgs_B.append(img_B)
        imgs_A = preprocess(np.array(imgs_A))
        imgs_B = preprocess(np.array(imgs_B))
        return imgs_A, imgs_B#加载验证集数据

    def load_batch(self, batch_size=1, data_crop=True, data_augment=True, gain=True, patch_size=(256,256)):
        self.n_batches = self.num_train//batch_size
        for i in range(self.n_batches-1):
            batch_A = self.trainA_paths[i*batch_size:(i+1)*batch_size]
            batch_B = self.trainB_paths[i*batch_size:(i+1)*batch_size]
            imgs_A, imgs_B = [], []
            for idx in range(len(batch_A)):
                img_A, img_B = read_pair(batch_A[idx], batch_B[idx])
                if (data_crop):
                    img_A, img_B = random_crop(img_A, img_B, patch_size)
                if (gain):
                    img_A, img_B = gain_ratio(img_A, img_B)
                if (data_augment):
                    img_A, img_B = augment(img_A, img_B)
                # if (gain):
                #     img_A, img_B = gain_ratio(img_A, img_B)
                imgs_A.append(img_A)
                imgs_B.append(img_B)
            imgs_A = preprocess(np.array(imgs_A))
            imgs_B = preprocess(np.array(imgs_B))
            yield imgs_A, imgs_B#加载训练集
