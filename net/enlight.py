import tensorflow as tf


class BrightnessEnhancementNet(tf.keras.Model):
    def __init__(self):
        super(BrightnessEnhancementNet, self).__init__()

        # 卷积层
        self.conv1 = tf.keras.layers.Conv2D(32, kernel_size=3, strides=1, padding='same')
        self.conv2 = tf.keras.layers.Conv2D(64, kernel_size=3, strides=1, padding='same')

        # ReLU激活函数
        self.relu = tf.keras.layers.ReLU()

        # 上采样层
        self.upsample = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.upsample(x)

        return x