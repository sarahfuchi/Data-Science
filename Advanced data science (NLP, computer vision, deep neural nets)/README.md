![I am something of a painter myself](/images/monet/monet0.jpg)

“Every artist dips his brush in his own soul, and paints his own nature into his pictures.”
-Henry Ward Beecher

This project uses Kaggle datasets and gets inspiration from public notebooks.

# Table of Contents
1. [Chapter 1 - Project Overview](#ch1)
1. [Chapter 2 - Data Science Steps](#ch2)
1. [Chapter 3 - Step 1: Problem Definition](#ch3)
1. [Chapter 4 - Step 2: Data Gathering](#ch4)
1. [Chapter 5 - Step 3: Data Preparation](#ch5)
1. [Chapter 6 - Step 4: Build the Generator](#ch6)
1. [Chapter 7 - Step 5: Build the Discriminator](#ch7)
1. [Chapter 8 - Step 6: Build the CycleGAN Model](#ch8)
1. [Chapter 9 - Step 7: Define the loss functions](#ch9)
1. [Chapter 10 - Step 8: Train the CycleGAN](#ch10)
1. [Chapter 11 - Step 9: Visualization](#ch11)


1. [References](#ch90)


<a id="ch1"></a>
# Project Overview
I have always enjoyed painting and working with data. I have two passions that I want to merge together and do something creative with them. Painters, such as Claude Monet, have unique brush strokes and color choices that set them apart from other painters.The project revolves around using Generative Adversarial Networks (GAN) to incorporate Monet's style into existing photographs and recreate that style from scratch.

If my photos are deemed successful by the classifier, then I will be considered a junior Monet. I'm confident that the computer vision technology we're using now can handle this task, so let's see how well it performs Monetization-wise.

**A GAN** can be formed from two types of neural networks: a generator model and a discriminator model. The generator produces the images. For this project, I created images in the style of Monet. This machine learning algorithm is used to differentiate between real images and generated ones.The two models will compete against each other, with the generator trying to convince the discriminator, and the discriminator trying to accurately identify real images from generated ones. I have created a GAN that can generate images similar to those created by Monet.

Let's take a look at the steps:  

<a id="ch2"></a>
# Data Science Steps
1. **Problem Definition:** Is the image real or is it generated through classification? 
2. **Data Gathering:** I used the Monet TFRecord dataset and the Photo TFRecord dataset from the KaggleDatasets app.
3. **Data Preperation:** I normalized and scaled the data before processing it.
4. **EDA (Explanatory Data Analysis):** It is important to use descriptive and graphical statistics to look for patterns, correlations, and comparisons in the dataset. In this step, I used visualization techniques to analyze the data.
5. **Data Modelling:** In this project, I developed a generator to produce Monet-style photos, as well as a discriminator to determine whether an image is real or fake. I used a CycleGAN architecture to train the model.
6. **Validate Model:** After training the model, I validated it to check the accuracy of the model I had built.
7. **Optimize Model:** The loss function was used to adjust the weighting of the model in order to optimize the prediction results. 

<a id="ch3"></a>
# Step 1: Problem Definition
The goal of this project is to create Monet-like photos either from scratch or from existing photos, and then to classify them according to whether they are real or fake.

**Project Summary from Kaggle:**
We appreciate the works of artists by looking at their unique style, such as the colors they choose or the way they brush their strokes. Thanks to Generative Adversarial Networks (GANs), it is now possible to algorithmically imitate the "je ne sais quoi" of artists like Claude Monet.In this project, I will be able to showcase my creative photography style or create a new style that reflects my personality.

Computer vision has improved a lot in recent years, and GANs can now create realistic images of objects quite convincingly. Some people believe that creating masterpieces that are considered museum-worthy is more an art than a science. So can (data) science, in the form of GAN, trick classifiers into believing that they have created true Monet? This is the challenge I will accept!

The Challenge:
A GAN can include a generator model and a discriminator model. The generator creates the images. For my project, I will create images in the style of Monet. This generator has been trained to differentiate between different types of data.

The two models will compete against each other, with the generator trying to trick the discriminator, and the discriminator trying to correctly identify the real images from the generated ones. The task is to build a GAN that can generate images similar to Monet's.


<a id="ch4"></a>
# Step 2: Data Gathering

The dataset for this project can be found on the Kaggle website: [Kaggle: I’m Something of a Painter Myself](https://www.kaggle.com/competitions/gan-getting-started/data) or using the Kaggle app in Python. I chose the second option. 

<a id="ch5"></a>
# Step 3: Data Preperation
The data was already pre-processed when it came from Kaggle, so I just focused on scaling and normalizing it further. All the images were already sized to 256x256 pixels. I also scaled the images to a [-1,1] scale. Since we're building a generative model, we don't need labels or image ids in this project.

## 3.1 Import Libraries

```
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
from kaggle_datasets import KaggleDatasets
import matplotlib.pyplot as plt
import numpy as np
```

## 3.2 Pre-view of the Data

The dataset contained four directories: money_tfrec, photo_tfrec, money_jpg, and photo_jpg. The monet_tfrec and monet_jpg directories contained the same painting and photo images, respectively. I followed the TFRecords dataset. The monet directories contained paintings of Monet. I used these images to train my model.

The photo directories contained photos. I modified the images to have the same style as Monet using machine learning algorithms (GAN architectures). The CycleGAN dataset includes other artists' styles as well.

All of the images for the project were already in a 256x256 resolution. To ensure that the images are in RGB format, I set the color channel to 3. I needed to scale the images to [-1, 1] so that they would be the same size. Since the model is a generative model, I don't need to use the labels or image id to get the image, I'll just get the image from the TFRecord.

```
IMAGE_SIZE = [256, 256]

def decode_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = (tf.cast(image, tf.float32) / 127.5) - 1
    image = tf.reshape(image, [*IMAGE_SIZE, 3])
    return image

def read_tfrecord(example):
    tfrecord_format = {
        "image_name": tf.io.FixedLenFeature([], tf.string),
        "image": tf.io.FixedLenFeature([], tf.string),
        "target": tf.io.FixedLenFeature([], tf.string)
    }
    example = tf.io.parse_single_example(example, tfrecord_format)
    image = decode_image(example['image'])
    return image
```

**Files**
- monet_jpg - 300 Monet paintings sized 256x256 in JPEG format
- monet_tfrec - 300 Monet paintings sized 256x256 in TFRecord format
- photo_jpg - 7028 photos sized 256x256 in JPEG format
- photo_tfrec - 7028 photos sized 256x256 in TFRecord format

![photo_and_monet_example.jpg](/images/monet/monet1.jpg)

<a id="ch5"></a>
## 3.3 Data Pre-processing: 

I used the UNET architecture for CycleGAN. In order to build the generator, upsampling and downsampling methods are needed. More information on upsampling and downsampling will be discussed below.
The downsample reduces the width and height of the image by the stride. The stride is the length of the step the filter takes. Since the stride is two, the filter was applied to every other pixel, resulting in a reduction of weight and height by two.

Normalization or feature scaling is a technique that helps to ensure that features with very diverse ranges will have an equal impact on network performance. Without normalization, some functions or variables may be ignored. In this particular project, I used instance normalization (IN), which operates on individual samples, rather than batch normalization (BN).Both normalization methods can speed up training and make the network converge faster. There is a main difference between the two methods, IN transforming a single training sample, while BN does that to the whole mini-batch of samples. [More information can be found here on IN and BNs.](https://www.baeldung.com/cs/instance-vs-batch-normalization) 

![instance_and_batch_normalization.jpg](/images/monet/monet2.jpg)

```
OUTPUT_CHANNELS = 3

def downsampler(filters, size, apply_instancenorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)
    gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

    result = keras.Sequential()
    result.add(layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

    if apply_instancenorm:
        result.add(tfa.layers.InstanceNormalization(gamma_initializer=gamma_init))

    result.add(layers.LeakyReLU())

    return result
```

Upsampling is the opposite of downsampling and increases the size of the image. [There is a good article on upsampling and downsampling available here.](https://medium.com/analytics-vidhya/downsampling-and-upsampling-of-images-demystifying-the-theory-4ca7e21db24a) 

I prefer visuals to explanations, so I can provide a visual example of upsampling:

![upsample_example.jpg](/images/monet/monet3.jpg)

```
def upsampler(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)
    gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

    result = keras.Sequential()
    result.add(layers.Conv2DTranspose(filters, size, strides=2,
                                      padding='same',
                                      kernel_initializer=initializer,
                                      use_bias=False))

    result.add(tfa.layers.InstanceNormalization(gamma_initializer=gamma_init))

    if apply_dropout:
        result.add(layers.Dropout(0.5))

    result.add(layers.ReLU())

    return result
```


<a id="ch6"></a>
# Step 4: Build the Generator

The way I have implemented the code is by downsampling the input image first, and then upsampling it while establishing long skip connections.

```
def Generator():
    inputs = layers.Input(shape=[256,256,3])

    # bs = batch size
    down_stack = [
        downsampler(64, 4, apply_instancenorm=False), 
        downsampler(128, 4), 
        downsampler(256, 4), 
        downsampler(512, 4), 
        downsampler(512, 4), 
        downsampler(512, 4), 
        downsampler(512, 4), 
        downsampler(512, 4), 
    ]

    up_stack = [
        upsampler(512, 4, apply_dropout=True), 
        upsampler(512, 4, apply_dropout=True), 
        upsampler(512, 4, apply_dropout=True), 
        upsampler(512, 4), 
        upsampler(256, 4), 
        upsampler(128, 4), 
        upsampler(64, 4), 
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                  strides=2,
                                  padding='same',
                                  kernel_initializer=initializer,
                                  activation='tanh') # (bs, 256, 256, 3)

    x = inputs

    # Downsampling 
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling followed by the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = layers.Concatenate()([x, skip])

    x = last(x)

    return keras.Model(inputs=inputs, outputs=x)
```


<a id="ch7"></a>
# Step 5: Build the Discriminator

The discriminator recognizes and classifies images as either real or fake. The fake image I am referring to is the image generated by the generator, not the genuine Monet image. The discriminator outputs a smaller image which is more accurate in determining whether the object is real or fake.

```
def Discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)
    gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

    inp = layers.Input(shape=[256, 256, 3], name='input_image')

    x = inp

    down1 = downsampler(64, 4, False)(x) 
    down2 = downsampler(128, 4)(down1) 
    down3 = downsampler(256, 4)(down2) 

    zero_pad1 = layers.ZeroPadding2D()(down3) 
    conv = layers.Conv2D(512, 4, strides=1,
                         kernel_initializer=initializer,
                         use_bias=False)(zero_pad1) 

    norm1 = tfa.layers.InstanceNormalization(gamma_initializer=gamma_init)(conv)

    leaky_relu = layers.LeakyReLU()(norm1)

    zero_pad2 = layers.ZeroPadding2D()(leaky_relu) 

    last = layers.Conv2D(1, 4, strides=1,
                         kernel_initializer=initializer)(zero_pad2) 

    return tf.keras.Model(inputs=inp, outputs=last)
```
```
with strategy.scope():
    monet_generator = Generator() # transforms photos into paintings that look like the work of Monet.
    photo_generator = Generator() # transforms Monet paintings into photos.

    monet_discriminator = Discriminator() # differentiates features between real Monet paintings and generated Monet paintings.
    photo_discriminator = Discriminator() # differentiates features between real photos and generated photos.
```
At this point, our generators are not yet trained, so the generated Monet-esque photo does not look like what we expect.

```
to_monet = monet_generator(example_photo)

plt.subplot(1, 2, 1)
plt.title("Original Photo Example")
plt.imshow(example_photo[0] * 0.5 + 0.5)

plt.subplot(1, 2, 2)
plt.title("Monet-esque Photo Example")
plt.imshow(to_monet[0] * 0.5 + 0.5)
plt.show()
```
![generated_Monet.jpg](/images/monet/monet4.jpg)

<a id="ch8"></a>
# Step 6: Build the CycleGAN Model

In this section, I created a new subclass of the tf.keras.Model class. The idea is to use the fit() function to train the model later. During the training step, the model can change a photo into a painting by transforming it using a Monet algorithm. Later, the model can revert the painting back to a photo.The difference between the original image and the image converted twice is the loss of cycle consistency. The expectation is that the original photo and the twice-transformed photo will share similar features. A simple translation of Cycle GAN can be seen in the image below, inspired by the work of Aamir Jarda. [More can be seen here: A Gentle Introduction to Cycle Consistent Adversarial Networks article.](https://towardsdatascience.com/a-gentle-introduction-to-cycle-consistent-adversarial-networks-6731c8424a87)

![translation_cycle.jpg](/images/monet/monet5.jpg)

```
class CycleGan(keras.Model):
    def __init__(
        self,
        monet_generator,
        photo_generator,
        monet_discriminator,
        photo_discriminator,
        lambda_cycle=10,
    ):
        super(CycleGan, self).__init__()
        self.m_gen = monet_generator
        self.p_gen = photo_generator
        self.m_disc = monet_discriminator
        self.p_disc = photo_discriminator
        self.lambda_cycle = lambda_cycle
        
    def compile(
        self,
        m_gen_optimizer,
        p_gen_optimizer,
        m_disc_optimizer,
        p_disc_optimizer,
        gen_loss_fn,
        disc_loss_fn,
        cycle_loss_fn,
        identity_loss_fn
    ):
        super(CycleGan, self).compile()
        self.m_gen_optimizer = m_gen_optimizer
        self.p_gen_optimizer = p_gen_optimizer
        self.m_disc_optimizer = m_disc_optimizer
        self.p_disc_optimizer = p_disc_optimizer
        self.gen_loss_fn = gen_loss_fn
        self.disc_loss_fn = disc_loss_fn
        self.cycle_loss_fn = cycle_loss_fn
        self.identity_loss_fn = identity_loss_fn
        
    def train_step(self, batch_data):
        real_monet, real_photo = batch_data
        
        with tf.GradientTape(persistent=True) as tape:
            # photo --> monet --> photo
            fake_monet = self.m_gen(real_photo, training=True)
            cycled_photo = self.p_gen(fake_monet, training=True)

            # monet --> photo --> monet
            fake_photo = self.p_gen(real_monet, training=True)
            cycled_monet = self.m_gen(fake_photo, training=True)

            # generating itself
            same_monet = self.m_gen(real_monet, training=True)
            same_photo = self.p_gen(real_photo, training=True)

            # discriminator (input: real images)
            disc_real_monet = self.m_disc(real_monet, training=True)
            disc_real_photo = self.p_disc(real_photo, training=True)

            # discriminator (input: fake images)
            disc_fake_monet = self.m_disc(fake_monet, training=True)
            disc_fake_photo = self.p_disc(fake_photo, training=True)

            # evaluates generator loss
            monet_gen_loss = self.gen_loss_fn(disc_fake_monet)
            photo_gen_loss = self.gen_loss_fn(disc_fake_photo)

            # evaluates total cycle consistency loss
            total_cycle_loss = self.cycle_loss_fn(real_monet, cycled_monet, self.lambda_cycle) + self.cycle_loss_fn(real_photo, cycled_photo, self.lambda_cycle)

            # evaluates total generator loss
            total_monet_gen_loss = monet_gen_loss + total_cycle_loss + self.identity_loss_fn(real_monet, same_monet, self.lambda_cycle)
            total_photo_gen_loss = photo_gen_loss + total_cycle_loss + self.identity_loss_fn(real_photo, same_photo, self.lambda_cycle)

            # evaluates discriminator loss
            monet_disc_loss = self.disc_loss_fn(disc_real_monet, disc_fake_monet)
            photo_disc_loss = self.disc_loss_fn(disc_real_photo, disc_fake_photo)

        # Calculate the gradients for generator and discriminator
        monet_generator_gradients = tape.gradient(total_monet_gen_loss,
                                                  self.m_gen.trainable_variables)
        photo_generator_gradients = tape.gradient(total_photo_gen_loss,
                                                  self.p_gen.trainable_variables)

        monet_discriminator_gradients = tape.gradient(monet_disc_loss,
                                                      self.m_disc.trainable_variables)
        photo_discriminator_gradients = tape.gradient(photo_disc_loss,
                                                      self.p_disc.trainable_variables)

        # Apply the gradients to the optimizer
        self.m_gen_optimizer.apply_gradients(zip(monet_generator_gradients,
                                                 self.m_gen.trainable_variables))

        self.p_gen_optimizer.apply_gradients(zip(photo_generator_gradients,
                                                 self.p_gen.trainable_variables))

        self.m_disc_optimizer.apply_gradients(zip(monet_discriminator_gradients,
                                                  self.m_disc.trainable_variables))

        self.p_disc_optimizer.apply_gradients(zip(photo_discriminator_gradients,
                                                  self.p_disc.trainable_variables))
        
        return {
            "monet_gen_loss": total_monet_gen_loss,
            "photo_gen_loss": total_photo_gen_loss,
            "monet_disc_loss": monet_disc_loss,
            "photo_disc_loss": photo_disc_loss
        }
```

<a id="ch9"></a>
# Step 7: Define the Loss Functions

The discriminator loss function computes the difference between a real image and a matrix of 1s, and between a fake image and a matrix of 0s. A perfect discriminator will output all 1s for real images and all 0s for fake images. The discriminator loss outputs the average of the true loss and the generated loss.

```
with strategy.scope():
    def discriminator_loss(real, generated):
        real_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(tf.ones_like(real), real)

        generated_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(tf.zeros_like(generated), generated)

        total_disc_loss = real_loss + generated_loss

        return total_disc_loss * 0.5
```
The generator wants to convince the discriminator into thinking the generated image is real. The perfect generator will have the discriminator output only 1s. Thus, it compares the generated image to a matrix of 1s to find the loss.
```
with strategy.scope():
    def generator_loss(generated):
        return tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(tf.ones_like(generated), generated)
```
The generator is trying to convince the discriminator that the generated image is real. A generator that produces only 1s as its discriminator output would be ideal. The loss is determined by comparing the generated image to a matrix of 1s to find the difference.
```
with strategy.scope():
    def calc_cycle_loss(real_image, cycled_image, LAMBDA):
        loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))

        return LAMBDA * loss1
```
The identity loss compares the image with the source from which it was generated. We want an image to be generated that is identical to the original image. The identity loss compares the input with the output of the generator.

```
with strategy.scope():
    def identity_loss(real_image, same_image, LAMBDA):
        loss = tf.reduce_mean(tf.abs(real_image - same_image))
        return LAMBDA * 0.5 * loss
```
<a id="ch10"></a>
# Step 8: Train the CycleGAN
In this part of the project, I compiled the model. Since I used tf.keras.Model to build the CycleGAN, now is the time to use the fit() function to train.

```
with strategy.scope():
    monet_generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    photo_generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    monet_discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    photo_discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
```
```
with strategy.scope():
    cycle_gan_model = CycleGan(
        monet_generator, photo_generator, monet_discriminator, photo_discriminator
    )

    cycle_gan_model.compile(
        m_gen_optimizer = monet_generator_optimizer,
        p_gen_optimizer = photo_generator_optimizer,
        m_disc_optimizer = monet_discriminator_optimizer,
        p_disc_optimizer = photo_discriminator_optimizer,
        gen_loss_fn = generator_loss,
        disc_loss_fn = discriminator_loss,
        cycle_loss_fn = calc_cycle_loss,
        identity_loss_fn = identity_loss
    )
```
```
cycle_gan_model.fit(
    tf.data.Dataset.zip((monet_ds, photo_ds)),
    epochs=25
)
```
![epochs.jpg](/images/monet/monet6.jpg)

<a id="ch11"></a>
# Step 9: Visualization 

Now is the time to see how the algorithm translated the photos in to Monet-Esque: 

```
_, ax = plt.subplots(5, 2, figsize=(12, 12))
for i, img in enumerate(photo_ds.take(5)):
    prediction = monet_generator(img, training=False)[0].numpy()
    prediction = (prediction * 127.5 + 127.5).astype(np.uint8)
    img = (img[0] * 127.5 + 127.5).numpy().astype(np.uint8)

    ax[i, 0].imshow(img)
    ax[i, 1].imshow(prediction)
    ax[i, 0].set_title("Input Photo Example")
    ax[i, 1].set_title("Monet-esque Example")
    ax[i, 0].axis("off")
    ax[i, 1].axis("off")
plt.show()
```
![monetesque.jpg](/images/monet/monet7.jpg)

<a id="ch90"></a>
# References
I would like to express gratitude for the following resources, and thank developers for the inspiration:

* [Monet CycleGAN Tutorial](https://www.kaggle.com/code/amyjang/monet-cyclegan-tutorial) - Indepth dive to Cyclegan steps.
* [Instance vs Batch Normalization](https://www.baeldung.com/cs/instance-vs-batch-normalization) -  Normalization (IN) and Batch Normalization (BN) overview.
* [Downsampling and Upsampling of Images](https://medium.com/analytics-vidhya/downsampling-and-upsampling-of-images-demystifying-the-theory-4ca7e21db24a) - Downsampling and Upsampling of Images - Demystifying the Theory.
* [A Gentle Introduction to Cycle Consistent Adversarial Networks](https://towardsdatascience.com/a-gentle-introduction-to-cycle-consistent-adversarial-networks-6731c8424a87) - Article going over what exactly Cycle GAN are and what are the existing applications of such models are.




