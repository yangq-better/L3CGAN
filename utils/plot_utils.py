import os
import matplotlib.pyplot as plt
from PIL import Image
from os.path import join
import  numpy as np

def save_val_samples_funieGAN(samples_dir, dis_imgs, gen_imgs, gt_imgs, step, N_samples=3):
    dis_imgs = dis_imgs*255
    gen_imgs = np.minimum(gen_imgs, 1)*255#让像素不超过255
    gt_imgs = gt_imgs*255
    for i in range(N_samples):
        dis_img = dis_imgs[i,:,:,:].astype('uint8')
        gen_img = gen_imgs[i,:,:,:].astype('uint8')
        gt_img = gt_imgs[i,:,:,:].astype('uint8')
        save_img = np.hstack((dis_img, gen_img, gt_img))
        Image.fromarray(save_img).save(join(samples_dir,("%d_%d.png" %(step,i))))
def draw_and_save_loss(samples_dir, all_losses, iters, step):
    # plt.figure()
    plt.plot(iters, all_losses, label="gloss")
    # plt.draw()
    # plot.show()
    plt.savefig(os.path.join(samples_dir, ("%d.png" %step)))
def draw_and_save_loss2(samples_dir, all_g_losses, all_d_losses, iters, step):
    # plt.figure()
    plt.plot(iters, all_g_losses, 'g', label='g_loss')#
    plt.plot(iters, all_d_losses, 'r', label='d_loss')#
    plt.legend(['g_loss','d_loss'])

    plt.savefig(os.path.join(samples_dir, ("%d.png" % step)))
def save_val_samples_unpaired(samples_dir, gen_imgs, step, N_samples=1, N_ims=6):
    row=2*N_samples; col=N_ims//2;
    titles = ['Original','Translated','Reconstructed']
    fig, axs = plt.subplots(row, col)
    cnt = 0
    for i in range(row):
        for j in range(col):
            axs[i,j].imshow(gen_imgs[cnt])
            axs[i, j].set_title(titles[j])
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig(os.path.join(samples_dir, ("_%d.png" %step)))
    plt.close()
def save_test_samples_funieGAN(samples_dir, gen_imgs, step=0):
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(gen_imgs[0])
    axs[0].set_title("Input")
    axs[0].axis('off')
    axs[1].imshow(gen_imgs[1])
    axs[1].set_title("Generated")
    axs[1].axis('off')
    fig.savefig(os.path.join(samples_dir,("_test_%d.png" %step)))
    plt.close()
def viz_gen_and_dis_losses(all_D_losses, all_G_losses, save_dir=None):
    plt.plot(all_D_losses, 'r')
    plt.plot(all_G_losses, 'g')
    plt.title('Model convergence'); plt.ylabel('Losses'); plt.xlabel('# of steps');
    plt.legend(['Discriminator network', 'Generator network'], loc='upper right')
    plt.show();
    if not save_dir:
        plt.savefig(os.path.join(save_dir, '_conv.png'))