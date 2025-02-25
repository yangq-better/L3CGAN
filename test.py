import os
import time
import ntpath
import numpy as np
from PIL import Image
import scipy.misc
from os.path import join, exists
from keras.models import model_from_json
## local libs
import random
from utils.data_utils import getPaths, read, preprocess, deprocess

os.environ["CUDA_VISIBLE_DEVICES"]="-1" #禁用GPU
import tensorflow as tf
tf.config,set_visible_devices([], 'GPU') #禁用GPU,确保Keras使用CPU

## for testing arbitrary local data用于测试任意的本地数据
data_dir = "./data/Dataset_test/LOLDataset//eval15/low/"
gt_gata_dir = "./data/Dataset_test/LOLDataset//eval15/high/"
from utils.data_utils import get_local_test_data
test_paths = getPaths(data_dir)
print ("{0} test images are loaded".format(len(test_paths)))

## create dir for log and (sampled) validation data
samples_dir = "./output_data/output/"
if not exists(samples_dir): os.makedirs(samples_dir)
samples_dir_0 = "./output_data/output_gen/"
if not exists(samples_dir_0): os.makedirs(samples_dir_0)
## test ganbefore
#checkpoint_dir  = 'checkpoints/CGAN/our485/'#若要使用重新训练的结果，请取消此行注释，然后注释下一行
checkpoint_dir  = 'checkpoints_pretrained/'#使用本文训练出的结果
model_name_by_epoch = "model_485000_"

model_h5 = checkpoint_dir + model_name_by_epoch + ".h5"
model_json = checkpoint_dir + model_name_by_epoch + ".json"
# sanity
assert (exists(model_h5) and exists(model_json))

# load model
with open(model_json, "r") as json_file:
    loaded_model_json = json_file.read()
funie_gan_generator = model_from_json(loaded_model_json)
# load weights into new model
funie_gan_generator.load_weights(model_h5)#.encode('utf8')
print("\nLoaded data and model")

# testing loop
times = []; #s = time.time()
for img_path in test_paths:
    # prepare data
    img_name = ntpath.basename(img_path)
    gt_img = read(join(gt_gata_dir, img_name))
    inp_img = read(img_path)

    # inp_img = inp_img[:, 0:592, :]
    ratio = int(np.mean(gt_img)/np.mean(inp_img))
    # print(ratio)
    inp_img_ratio = inp_img*ratio
    inp_img_ = inp_img*ratio#[0:256,0:256,:]
    inp_img_ratio = np.minimum(inp_img_ratio, 255)
    inp_img_ = np.minimum(inp_img_, 255)#inp_img*255/np.max(np.max(inp_img, axis=-1))

    im = preprocess(inp_img_)
    im = np.expand_dims(im, axis=0)
    # generate enhanced image
    s = time.time()
    gen = funie_gan_generator.predict(im)
    gen_img = deprocess(np.minimum(gen,1))[0]
    # gen_img = np.minimum(gen_img,1)
    tot = time.time()-s
    times.append(tot)
    # save output images
    out_img = np.hstack((inp_img, inp_img_ratio, gen_img, gt_img)).astype('uint8')#[0:256,0:256,:][0:256,0:256,:]
    Image.fromarray(out_img).save(join(samples_dir, img_name))
    Image.fromarray(gen_img).save(join(samples_dir_0, img_name))


# some statistics
num_test = len(test_paths)
if (num_test==0):
    print ("\nFound no images for test")
else:
    print ("\nTotal images: {0}".format(num_test))
    # accumulate frame processing times (without bootstrap)
    Ttime, Mtime = np.sum(times[1:]), np.mean(times[1:])
    print ("Time taken: {0} sec at {1} fps".format(Ttime, 1./Mtime))
    print("\nSaved generated images in in {0}\n".format(samples_dir))

