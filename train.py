import os
import numpy as np
from os.path import  join, exists
from net.ganbefore import FUNIE_GAN
from utils.data_utils import DataLoader
from utils.plot_utils import save_val_samples_funieGAN, draw_and_save_loss2

os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

data_dir= "./data/Dataset_train/LOLDataset"
dataset_name = "our485"
data_loader = DataLoader(join(data_dir, dataset_name), dataset_name)

samples_dir = join("data/samples/CGAN/", dataset_name)#创建文件夹
checkpoint_dir = join("checkpoints/CGAN/", dataset_name)#创建文件夹
if not exists(samples_dir): os.makedirs(samples_dir)
if not exists(checkpoint_dir): os.makedirs(checkpoint_dir)

## 训练参数设定
patch_size = (256, 256)#256
num_epoch = 4000
batch_size = 4
val_interval = 500
N_val_samples = 3
save_model_interval = 2000
num_step = num_epoch*data_loader.num_train//batch_size

## load model arch
funie_gan = FUNIE_GAN(imrow = patch_size[0], imcol = patch_size[1])
valid = np.ones((batch_size,) + funie_gan.disc_patch)#
fake = np.zeros((batch_size,) + funie_gan.disc_patch)
## training loop
step = 0
all_G_losses = []
all_D_losses = []
iters = []
steps = 0

while (step <= num_step):
    for _, (imgs_distorted, imgs_good) in enumerate(data_loader.load_batch(batch_size,patch_size=patch_size)):
        ##  train the discriminator
        imgs_fake = funie_gan.generator.predict(imgs_distorted)
                ## train the generator
        g_loss = funie_gan.combined.train_on_batch([imgs_good,imgs_distorted], [valid,imgs_good])
        ## increment step, save losses, and print them
        step += 1 # all_D_losses.append(d_loss[0]);  ;

        if step % 10 == 0:
            d_loss_real = funie_gan.discriminator.train_on_batch([imgs_good,imgs_distorted], valid)  # 
            d_loss_fake = funie_gan.discriminator.train_on_batch([imgs_fake,imgs_distorted], fake)  #
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        if step % 50 == 0:
            print ("Step {0}/{1}: lossG: {2}, lossG_mse: {3}, lossG_cus: {4}, lossD:{5}".format(step, num_step, g_loss[0],g_loss[1],g_loss[2], d_loss))
        ## validate and save generated samples at regular intervals  d_loss[0],
        if (step % val_interval==0):
            imgs_distorted, imgs_good = data_loader.load_val_data(batch_size=N_val_samples,patch_size=patch_size)
            imgs_fake = funie_gan.generator.predict(imgs_distorted)
            save_val_samples_funieGAN(samples_dir, imgs_distorted, imgs_fake, imgs_good, step, N_samples=N_val_samples)
        ## save model and weights
        if (step % save_model_interval==0 or step == num_step):
            model_name = join(checkpoint_dir, ("model_%d" %step))
            with open(model_name+"_.json", "w") as json_file:
                json_file.write(funie_gan.generator.to_json())#保存模型的结构
            funie_gan.generator.save_weights(model_name+"_.h5")
            print("\nSaved trained model in {0}\n".format(checkpoint_dir))
        ## sanity
        if (step>=num_step): break