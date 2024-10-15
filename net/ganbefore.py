from __future__ import print_function, division
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import scipy.signal
from PIL import Image
import numpy as np
import keras.backend as K
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input, Dropout, Concatenate, Lambda, Multiply, Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Convolution2DTranspose
from keras.layers import BatchNormalization, Activation, MaxPool2D, AveragePooling2D, Add, GlobalAveragePooling2D, Dense, Softmax, GlobalMaxPool2D
from keras.applications import  vgg19

def VGG19_Content(dataset = 'imagenet'):
    #如果没有输入dataset,则下载imagenet训练VGG
    vgg = vgg19.VGG19(include_top = False ,weights=dataset)#加载预训练权重
    vgg.trainable = False
    #content_layers = ['block5_conv2']
    content_outputs = vgg.get_layer('block5_conv2').output#改了
    return Model(vgg.input, content_outputs)#vgg.input

class FUNIE_GAN():
    def __init__(self, imrow=256, imcol=256, imchan=3, loss_meth='wgan'):
        self.img_rows, self.img_cols, self.channels = imrow, imcol, imchan
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)
        self.vgg_content = VGG19_Content()#返回了vgg19block5_conv2层的结构和input
        self.disc_patch = (int(imrow/16), int(imcol/16),1)#patch_gan
        self.gf, self.df = 32, 32# number of filters in the first layer of G and D G和D第一层的卷积核深度数量
        self.middle = 4
        self.out_dim = 7
        optimizer = Adam(0.0001, 0.9)
        optimizer_d = Adam(0.0000009, 0.9)#0.9是固定值，第一个参数是学习率
        self.discriminator = self.FUNIE_discriminator()#后面的判别器
        self.discriminator.compile(loss='mse', optimizer = optimizer_d)#参数设置
        self.generator = self.FUNIE_generator1()#只用了generator1
        fake_A = self.generator(img_B)#通过B生成fake_A
        self.discriminator.trainable = False#固定判别器
        valid = self.discriminator([fake_A, img_B])#判别器判别是否为真
        self.combined = Model(inputs = [img_A, img_B], outputs = [valid, fake_A])
        self.combined.compile(loss = ['mae', self.total_gen_loss], loss_weights=[0.2,0.8],optimizer = optimizer)
    def sobel_loss(self, y_true, y_pred):
        y_true_gray = tf.image.rgb_to_grayscale(y_true)
        y_pred_gray = tf.image.rgb_to_grayscale(y_pred)
        GX_arr = np.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
        GY_arr = np.array([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]])

        GX64 = tf.constant(GX_arr)
        GY64 = tf.constant(GY_arr)

        GX32 = tf.cast(GX64, dtype=tf.float32)
        GY32 = tf.cast(GY64, dtype=tf.float32)

        GX = tf.reshape(GX32, [3, 3, 1, 1])
        GY = tf.reshape(GY32, [3, 3, 1, 1])

        GXgEn = tf.nn.conv2d(y_pred_gray, GX, strides=[1, 1, 1, 1], padding='VALID')
        #GXgEn = scipy.signal.convolve(GX, y_pred_gray)
        GYgEn = tf.nn.conv2d(y_pred_gray, GY, strides=[1, 1, 1, 1], padding='VALID')
        #GYgEn = scipy.signal.convolve(GY, y_pred_gray)
        GXg = tf.nn.conv2d(y_true_gray, GX, strides=[1, 1, 1, 1], padding='VALID')
        #GXg = scipy.signal.convolve(GX, y_true_gray)
        GYg = tf.nn.conv2d(y_true_gray, GY, strides=[1, 1, 1, 1], padding='VALID')
        #GYg = scipy.signal.convolve(GY, y_true_gray)
        resgEn = tf.norm(GXgEn, ord=1, axis=-1) + tf.norm(GYgEn, ord=1, axis=-1)
        resg = tf.norm(GXg, ord=1, axis=-1) + tf.norm(GYg, ord=1, axis=-1)
        result = K.mean(tf.norm((resgEn - resg), ord=1, axis=-1))
        return  result#新加的loss

    def prewitt_loss(self, y_true, y_pred):
        y_true_gray = tf.image.rgb_to_grayscale(y_true)
        y_pred_gray = tf.image.rgb_to_grayscale(y_pred)
        GX_arr = np.array([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -1.0, -1.0]])
        GY_arr = np.array([[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]])
        GX64 = tf.constant(GX_arr)
        GY64 = tf.constant(GY_arr)

        GX32 = tf.cast(GX64, dtype=tf.float32)
        GY32 = tf.cast(GY64, dtype=tf.float32)

        GX = tf.reshape(GX32, [3, 3, 1, 1])
        GY = tf.reshape(GY32, [3, 3, 1, 1])

        GXgEn = tf.nn.conv2d(y_pred_gray, GX, strides=[1, 1, 1, 1], padding='VALID')
        #GXgEn = scipy.signal.convolve(GX, y_pred_gray)
        GYgEn = tf.nn.conv2d(y_pred_gray, GY, strides=[1, 1, 1, 1], padding='VALID')
        #GYgEn = scipy.signal.convolve(GY, y_pred_gray)
        GXg = tf.nn.conv2d(y_true_gray, GX, strides=[1, 1, 1, 1], padding='VALID')
        #GXg = scipy.signal.convolve(GX, y_true_gray)
        GYg = tf.nn.conv2d(y_true_gray, GY, strides=[1, 1, 1, 1], padding='VALID')
        #GYg = scipy.signal.convolve(GY, y_true_gray)
        resgEn = tf.norm(GXgEn, ord=1, axis=-1) + tf.norm(GYgEn, ord=1, axis=-1)
        resg = tf.norm(GXg, ord=1, axis=-1) + tf.norm(GYg, ord=1, axis=-1)
        result = K.mean(tf.norm((resgEn - resg), ord=1, axis=-1))

        return result

    def YUV_loss(self, y_true, y_pred):#Y代表明亮度，应该要增加Y而不是只有UV
        r_true = y_true[:, :, :, 0]
        g_true = y_true[:, :, :, 1]
        b_true = y_true[:, :, :, 2]
        r_pred = y_pred[:, :, :, 0]
        g_pred = y_pred[:, :, :, 1]
        b_pred = y_pred[:, :, :, 2]

        Y = 0.299*r_true + 0.587*g_true + 0.114*b_true
        Y_En = 0.299*r_pred + 0.587*g_pred +0.114*b_pred
        U = -0.147*r_true - 0.289*g_true + 0.436*b_true
        U_En = -0.147*r_pred - 0.289*g_pred + 0.436*b_pred
        V = 0.615*r_true - 0.515*g_true - 0.100*b_true
        V_En = 0.615*r_pred - 0.515*g_pred -0.100*b_pred

        res_Y = tf.norm((Y_En - Y), ord=1, axis=-1)
        res_U = tf.norm((U_En - U), ord=1, axis=-1)
        res_V = tf.norm((V_En - V), ord=1, axis=-1)

        result = K.mean(res_Y + res_U + res_V)
        return result
    

    def SSIM_loss(self, org_content, gen_content):
        mu_x = tf.reduce_mean(org_content)
        mu_y = tf.reduce_mean(gen_content)
        sigma_x = tf.reduce_mean(tf.square(org_content - mu_x))
        sigma_y = tf.reduce_mean(tf.square(gen_content - mu_y))
        cov = tf.reduce_mean((org_content - mu_x)*(gen_content - mu_y))

        K1 = 0.01
        K2 = 0.03
        L = 1.0

        C1 = (K1 * L) ** 2
        C2 = (K2 * L) ** 2

        ssim = (2 * mu_x * mu_y + C1) * (2 * cov + C2) / ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2))


        return tf.reduce_mean(ssim)


    def total_gen_loss(self, org_content, gen_content):
        vgg_org_content = self.vgg_content(org_content)
        vgg_gen_content = self.vgg_content(gen_content)

        content_loss = K.mean(K.abs(vgg_org_content-vgg_gen_content), axis = -1)

        mae_gen_loss = K.mean(K.abs(org_content - gen_content))#主体中有maeloss此处注释

        yuv_loss = self.YUV_loss(org_content, gen_content)
        #sobel_loss = self.sobel_loss(org_content, gen_content)
        prewitt_loss = self.prewitt_loss(org_content, gen_content)
        gen_total_err = 0.15*content_loss + 0.55*yuv_loss + 0.3*prewitt_loss
        return gen_total_err

    def FUNIE_generator1(self):
        def slice(x, a):
            return x[:, a, :, :]
        def RDB(x0, filter_num):
            x1 = Conv2D(filter_num, kernel_size=3, strides=1, padding='same')(x0)
            x1 = LeakyReLU(alpha = 0.2)(x1)

            c1 = Concatenate(axis=-1)([x0, x1])

            x2 = Conv2D(filter_num, kernel_size=3, strides=1,padding='same')(c1)
            x2 = LeakyReLU(alpha = 0.2)(x2)

            c2 = Concatenate(axis=-1)([x0,x1,x2])

            x3 = Conv2D(filter_num,kernel_size=3, strides=1, padding='same')(c2)
            x3 = LeakyReLU(alpha= 0.2)(x3)

            c3 = Concatenate(axis=-1)([x0,x1,x2,x3])

            x4 = Conv2D(filter_num, kernel_size=3, strides=1, padding='same')(c3)
            x4 = LeakyReLU(alpha=0.2)(x4)

            c4 = Concatenate(axis=-1)([x0, x1, x2, x3, x4])

            a0 = Conv2D(filter_num, kernel_size=1, strides=1, padding='same')(c4)

            a0 = channel_attention(a0, 8)#网络结构更改更改成正常的RDB及增加了spatial_attention
            a1 = spatial_attention(a0)

            a2 = Add()([x0, a1])
            return a2
        def RRDB(x0, filter_num):

            # RRDB块
            rdb_out1 = RDB(x0, filter_num)
            rdb_out1 = Lambda(lambda x: x * 0.5)(rdb_out1)
            x1 = Add()([x0, rdb_out1])

            rdb_out2 = RDB(x1, filter_num)
            rdb_out2 = Lambda(lambda x: x * 0.5)(rdb_out2)
            x2 = Add()([x1, rdb_out2])

            rdb_out3 = RDB(x2, filter_num)
            rdb_out3 = Lambda(lambda x: x*0.5)(rdb_out3)
            x3 = Add()([x2, rdb_out3])
            x3 = Lambda(lambda x: x * 0.5)(x3)

            x = Add()([x3, x0])


            # 增加残差连接


            return x
        def spatial_attention(input_feature):
            avg_pool = Lambda(lambda x: K.mean(x, axis=3,keepdims=True))(input_feature)
            max_pool = Lambda(lambda x:K.max(x,axis=3,keepdims=True))(input_feature)
            concat = Concatenate(axis=3)([max_pool,avg_pool])
            sa_feature = Conv2D(1,(7,7),strides=1,padding='same')(concat)
            sa_feature = LeakyReLU(alpha = 0.05)(sa_feature)
            return  Multiply()([input_feature, sa_feature])
        def channel_attention(input_feature, ratio):
            channel = input_feature._keras_shape[-1]#读取最后一维的大小
            shared_layer_one = Dense(channel//ratio, activation = 'relu', kernel_initializer='he_normal',use_bias=True, bias_initializer='zeros')
            shared_layer_two = Dense(channel, kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')

            avg_pool = GlobalAveragePooling2D()(input_feature)
            avg_pool = shared_layer_one(avg_pool)
            avg_pool = shared_layer_two(avg_pool)

            max_pool = GlobalMaxPool2D()(input_feature)
            max_pool = shared_layer_one(max_pool)
            max_pool = shared_layer_two(max_pool)

            ca_feature = Add()([avg_pool, max_pool])
            ca_feature = Activation('sigmoid')(ca_feature)

            return Multiply()([input_feature, ca_feature])

        #输入的主体
        input_0 = Input(shape=self.img_shape)
        print(input_0)
        conv_layer_1 = Conv2D(filters=self.gf, kernel_size=3,padding='same')(input_0)
        conv_layer_1 = LeakyReLU(alpha=0.2)(conv_layer_1)

        conv_layer_2 = Conv2D(filters=self.gf, kernel_size=3,padding='same')(conv_layer_1)
        conv_layer_2 = LeakyReLU(alpha=0.2)(conv_layer_2)

        rdb_1 = RRDB(conv_layer_2, self.gf)
        rdb_2 = RRDB(rdb_1, self.gf)
        rdb_3 = RRDB(rdb_2, self.gf)

        concat_1 = Concatenate(axis=-1)([rdb_1, rdb_2, rdb_3])
        concat_1 = channel_attention(concat_1, 8)#没打完
        concat_2 = spatial_attention(concat_1)

        concat_conv_1 = Conv2D(filters=self.gf, kernel_size=1, strides=1, padding='same')(concat_2)
        conv_layer_3 = Conv2D(filters=self.gf, kernel_size=3, padding='same')(concat_conv_1)
        conv_layer_3 = LeakyReLU(alpha=0.2)(conv_layer_3)

        add_1 = Add()([conv_layer_1, conv_layer_3])
        output_img = Conv2D(filters=self.channels, kernel_size=3,padding='same', activation='sigmoid')(add_1)
        print(output_img)
        return Model(input_0, output_img)

    def FUNIE_discriminator(self):
        def slice(x, a):
            return x[:, a, :, :]
        def repet(x, a, b):
            return K.repeat_elements(x, rep=a, axis=b)
        def sk_layer(sk_conv1, sk_conv2, sk_conv3, middle, out_dim):
            sum_u = Add()([sk_conv1, sk_conv2, sk_conv3])
            squeeze = GlobalAveragePooling2D()(sum_u)
            squeeze = Reshape((1, 1, out_dim))(squeeze)
            z = Dense(units=middle, use_bias=True)(squeeze)
            z = Activation('relu')(z)
            a1 = Dense(units=out_dim, use_bias=True)(z)
            a2 = Dense(units=out_dim, use_bias=True)(z)
            a3 = Dense(units=out_dim, use_bias=True)(z)
            before_softmax = Concatenate(axis=1)([a1, a2, a3])
            after_softmax = Softmax(axis=1)(before_softmax)
            a1 = Lambda(slice, arguments={'a':0})(after_softmax)
            a1 = Reshape((1, 1, out_dim))(a1)
            a2 = Lambda(slice, arguments={'a':1})(after_softmax)
            a2 = Reshape((1, 1, out_dim))(a2)
            a3 = Lambda(slice, arguments={'a':2})(after_softmax)
            a3 = Reshape((1, 1, out_dim))(a3)

            select_1 = Multiply()([sk_conv1, a1])
            select_2 = Multiply()([sk_conv2, a2])
            select_3 = Multiply()([sk_conv3, a3])
            out = Add()([select_1, select_2, select_3])

            return out
        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)
        combined_imgs = Concatenate(axis=-1)([img_A, img_B])

        out_1 = Conv2D(filters=self.df, kernel_size=3, strides=2, padding='same', activation='relu')(combined_imgs)
        out_2 = Conv2D(filters=self.df*2, kernel_size=3, strides=2, padding='same', activation='relu')(out_1)
        out_3 = Conv2D(filters=self.df*4, kernel_size=3, strides=2, padding='same', activation='relu')(out_2)
        out_4 = Conv2D(filters=self.df*8, kernel_size=3, strides=2, padding='same', activation='relu')(out_3)

        sk_conv1 = Conv2D(filters=7, kernel_size=3, padding='same', activation='relu', name='sk_conv_1')(out_4)
        sk_conv2 = Conv2D(filters=7, kernel_size=5, padding='same', activation='relu', name='sk_conv_2')(out_4)
        sk_conv3 = Conv2D(filters=7, kernel_size=7, padding='same', activation='relu', name='sk_conv_3')(out_4)
        sk_out = sk_layer(sk_conv1, sk_conv2, sk_conv3, self.middle, self.out_dim)

        validity = Conv2D(1, kernel_size=3, strides=1, padding='same')(sk_out)

        print(img_A)
        print(validity)
        return Model([img_A, img_B], validity)




