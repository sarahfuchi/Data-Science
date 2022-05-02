![Let's look at the home prices!](/images/house/house0.jpg)

In this project, I will summarize my work using regression techniques to calculate home prices. 

This project uses Kaggle datasets and gets inspiration from public notebooks.

# Table of Contents
1. [Chapter 1 - Project Overview](#ch1)
1. [Chapter 2 - Data Science Steps](#ch2)
1. [Chapter 3 - Step 1: Problem Definition](#ch3)
1. [Chapter 4 - Step 2: Data Gathering](#ch4)
1. [Chapter 5 - Step 3: EDA and Data Preparation](#ch5)
1. [Chapter 6 - Step 4: Build the Models](#ch6)
1. [Chapter 7 - Step 5: Model Comparison](#ch7)


1. [References](#ch90)


<a id="ch1"></a>
# Project Overview
Housing has been super hot, especially for the last couple of years. Also, It has been one of my personal interests to work on a project to predict housing prices. I come across this project in Kaggle, in this project, there are 79 explanatory variables describing most aspects of residential homes in Ames, Iowa. The task is to estimate the final price of each home. 

Let's take a look at the steps:  

<a id="ch2"></a>
# Data Science Steps
1. **Problem Definition:** Finding the final price of homes. Since house price is a continues variable, this is a regression problem.
2. **Data Gathering:** I used the USA_Housing dataset, I got access to them through the [Kaggle: House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data).  
3. **Data Preperation:** I prepped the data by using scaling methods.
4. **EDA (Explanatory Data Analysis):** It is essential to use descriptive and graphical statistics to look for patterns, correlations and comparisons in the dataset. In this step I mostly used heatmaps and correlation matrix to analyze the data. 
5. **Data Modelling:** In this project, I used different linear regression methods including Linear Regression, Robust Regression, Ridge Regression, LASSO Regression, Elastic Net, Polynomial Regression, Stochastic Gradient Descent, Artficial Neural Network,  Random Forest Regressor and Support Vector Machine.
6. **Validate Model:** After training the model, I worked with cross validation techniques to validate the model.
7. **Optimize Model:** In this particular project, I did not focus on optimizing the model but used the models from sklearn with constant parameters. Parameter optimization will be a future improvement for me to work on. 

<a id="ch3"></a>
# Step 1: Problem Definition
Goal is to predict the home prices using variables.

**Project Summary from Kaggle:**
Ask a home buyer to describe their dream house, and they probably won't begin with the height of the basement ceiling or the proximity to an east-west railroad. But this playground competition's dataset proves that much more influences price negotiations than the number of bedrooms or a white-picket fence.

With 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa, this competition challenges you to predict the final price of each home.

Practice Skills: 

- Creative feature engineering 
- Advanced regression techniques like random forest and gradient boosting

<a id="ch4"></a>
# Step 2: Data Gathering

Dataset can be found at the Kaggle's mainpage for this project: [House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) or using the Kaggle app in Python. 

**This is the input data from Kaggle :**  ['USA_Housing.csv']

I used the USA_Housing dataset. The data contains the following columns:

- <b> Avg. Area Income </b>: Avg. Income of residents of the city house is located in.
- <b> Avg. Area House Age </b>: Avg Age of Houses in same city.
- <b> Avg. Area Number of Rooms </b>: Avg Number of Rooms for Houses in same city.
- <b> Avg. Area Number of Bedrooms </b>: Avg Number of Bedrooms for Houses in same city.
- <b> Area Population </b>: Population of city hou se is located in
- <b> Price </b>: Price that the house sold at.
- <b> Address </b>: Address for the house.

<a id="ch5"></a>
# Step 3: EDA and Data Preperation


## 3.1 Import Libraries

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import hvplot.pandas
%matplotlib inline
```
-------------------------

## 3.2 Pre-view of the Data

```
USAhousing = pd.read_csv('/kaggle/input/usa-housing/USA_Housing.csv')
USAhousing.head()
```

![data_head.jpg](/images/house/house1.jpg)

```
USAhousing.info()
```

![data_head.jpg](/images/house/house2.jpg)

As seen above, there is no null values in the dataset. 

<a id="ch5"></a>
## 3.3 Exploratory Data Analysis (EDA): 

I used pairplots and heatmaps to look at the data in more detail. 

![pair_plots.jpg](/images/house/house3.jpg)

![pair_plots.jpg](/images/house/house4.jpg)

## 3.3 Data Preperation: 

The Kaggle dataset came in a cleaned up form. So in this section, I have worked on splitting the data to prep for the data modelling part. Data is first split into an X array that contains the features to train on, and a y array with the target variable, Price. I got rid of the address column in this case as it contained only text. 

```
X = USAhousing[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms', 'Avg. Area Number of Bedrooms', 'Area Population']]
y = USAhousing['Price']
```
Then I splitted the data into test and train. Train set will be used to train the data, test will be used to test the model's performance. 

```
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

Standardizing the data is a very important step before making the model. So I used SatndardScaler() to standardize the data. 

```
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('std_scalar', StandardScaler())
])

X_train = pipeline.fit_transform(X_train)
X_test = pipeline.transform(X_test)
```


Then defined the functions for displaying the model performace. 

```
from sklearn import metrics
from sklearn.model_selection import cross_val_score

def cross_val(model):
    pred = cross_val_score(model, X, y, cv=10)
    return pred.mean()

def print_evaluate(true, predicted):  
    mae = metrics.mean_absolute_error(true, predicted)
    mse = metrics.mean_squared_error(true, predicted)
    rmse = np.sqrt(metrics.mean_squared_error(true, predicted))
    r2_square = metrics.r2_score(true, predicted)
    print('MAE:', mae)
    print('MSE:', mse)
    print('RMSE:', rmse)
    print('R2 Square', r2_square)
    print('__________________________________')
    
def evaluate(true, predicted):
    mae = metrics.mean_absolute_error(true, predicted)
    mse = metrics.mean_squared_error(true, predicted)
    rmse = np.sqrt(metrics.mean_squared_error(true, predicted))
    r2_square = metrics.r2_score(true, predicted)
    return mae, mse, rmse, r2_square
```

<a id="ch6"></a>
# Step 4: Build the Models

## 4.1 Linear Regression

- Linear Assumption: Linear regression assumes that the relationship between the input and output is linear. In most cases, the unprocessed data may not be linear in real life. There are techniques to make the relationship linear (e.g. log transform for an exponential relationship).
- Remove Noise: Outliers need to be removed from the dataset if possible for a cleaner input. 
- Remove Collinearity: Linear regression have a tendency to over-fit the data when input variables are highly correlated. This is mainly why I looked at the pairplots to see if there was anything I could remove, which I did not see. 
- Normal Distribution: Linear regression makes better predictions if the input and output variables have a normal (Gaussian) distribution. There are some options to transform the variables to make their distribution more normal.
- Rescaling : Linear regression will often make more reliable predictions if you rescale input variables using standardization or normalization.

sklearn libraty makes it quite simple to apply the models as such: 

```
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression(normalize=True)
lin_reg.fit(X_train,y_train)
```

I then visualized the True values vs Predicted Values 

![pair_plots.jpg](/images/house/house5.jpg)

Let's take a look at the residual histogram: 

![pair_plots.jpg](/images/house/house6.jpg)

Now it comes to the evaluation part. Here is a good article I found very useful: [what metrics to use when evaluatin the regression models](https://towardsdatascience.com/what-are-the-best-metrics-to-evaluate-your-regression-model-418ca481755b)

Mainly there are 3 metrics for model evaluation in regression:
1. R Square/Adjusted R Square : R Square measures how much variability in dependent variable can be explained by the model.
2. Mean Square Error(MSE)/Root Mean Square Error(RMSE): While R Square is a relative measure of how well the model fits dependent variables, Mean Square Error is an absolute measure of the goodness for the fit.
3. Mean Absolute Error(MAE): Mean Absolute Error(MAE) is similar to Mean Square Error(MSE). However, instead of the sum of square of error in MSE, MAE is taking the sum of the absolute value of error.

Next, I went and printed the evaluation metrics for the linear regression model. 

![linear_regression.jpg](/images/house/house7.jpg)

## 4.2 Robust Regression

As stated at [ROBUST REGRESSION resource by UCLA](https://stats.oarc.ucla.edu/r/dae/robust-regression/): Robust regression is an alternative to least squares regression when data are contaminated with outliers or influential observations, and it can also be used for the purpose of detecting influential observations. A common example to use robust estimation is when the data contain outliers. 

I used the Random sample consensus (RANSAC) model via sklearn. It is an iterative method to estimnate parameters of a mathematical model while treating the outliers with no influence. 

![robust_regression.jpg](/images/house/house7.jpg)

## 4.3 Ridge Regression

<a id="ch7"></a>
# Step 5: Build the Discriminator

The discriminator takes in the input image and classifies it as real or fake. The fake I am referring to here is the image that is generator by the generator, not the genuine Monet image. Instead of outputing a single node, the discriminator outputs a smaller 2D image with higher pixel values indicating a real classification and lower values indicating a fake classification.

```
def Discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)
    gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

    inp = layers.Input(shape=[256, 256, 3], name='input_image')

    x = inp

    down1 = downsample(64, 4, False)(x) # (bs, 128, 128, 64)
    down2 = downsample(128, 4)(down1) # (bs, 64, 64, 128)
    down3 = downsample(256, 4)(down2) # (bs, 32, 32, 256)

    zero_pad1 = layers.ZeroPadding2D()(down3) # (bs, 34, 34, 256)
    conv = layers.Conv2D(512, 4, strides=1,
                         kernel_initializer=initializer,
                         use_bias=False)(zero_pad1) # (bs, 31, 31, 512)

    norm1 = tfa.layers.InstanceNormalization(gamma_initializer=gamma_init)(conv)

    leaky_relu = layers.LeakyReLU()(norm1)

    zero_pad2 = layers.ZeroPadding2D()(leaky_relu) # (bs, 33, 33, 512)

    last = layers.Conv2D(1, 4, strides=1,
                         kernel_initializer=initializer)(zero_pad2) # (bs, 30, 30, 1)

    return tf.keras.Model(inputs=inp, outputs=last)
```
```
with strategy.scope():
    monet_generator = Generator() # transforms photos to Monet-esque paintings
    photo_generator = Generator() # transforms Monet paintings to be more like photos

    monet_discriminator = Discriminator() # differentiates real Monet paintings and generated Monet paintings
    photo_discriminator = Discriminator() # differentiates real photos and generated photos
```
Since our generators are not trained yet, the generated Monet-esque photo does not show what is expected at this point.

```
to_monet = monet_generator(example_photo)

plt.subplot(1, 2, 1)
plt.title("Original Photo")
plt.imshow(example_photo[0] * 0.5 + 0.5)

plt.subplot(1, 2, 2)
plt.title("Monet-esque Photo")
plt.imshow(to_monet[0] * 0.5 + 0.5)
plt.show()
```
![generated_Monet.jpg](/images/monet/monet4.jpg)

<a id="ch8"></a>
# Step 6: Build the CycleGAN Model

In this section, I subclassed a tf.keras.Model. The idea is then to apply the fit() later to train the model. During the training step, the model transforms a photo to a Monet painting and then back to a photo. The difference between the original photo and the twice-transformed photo is the cycle-consistency loss. The expectation is the original photo and the twice-transformed photo to be similar to one another. A simple translation of Cycle GAN can be seen in the below image, inspired by [A Gentle Introduction to Cycle Consistent Adversarial Networks article.](https://towardsdatascience.com/a-gentle-introduction-to-cycle-consistent-adversarial-networks-6731c8424a87)

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
            # photo to monet back to photo
            fake_monet = self.m_gen(real_photo, training=True)
            cycled_photo = self.p_gen(fake_monet, training=True)

            # monet to photo back to monet
            fake_photo = self.p_gen(real_monet, training=True)
            cycled_monet = self.m_gen(fake_photo, training=True)

            # generating itself
            same_monet = self.m_gen(real_monet, training=True)
            same_photo = self.p_gen(real_photo, training=True)

            # discriminator used to check, inputing real images
            disc_real_monet = self.m_disc(real_monet, training=True)
            disc_real_photo = self.p_disc(real_photo, training=True)

            # discriminator used to check, inputing fake images
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

The discriminator loss function below compares real images to a matrix of 1s and fake images to a matrix of 0s. The perfect discriminator will output all 1s for real images and all 0s for fake images. The discriminator loss outputs the average of the real and generated loss.

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
The goal is our original photo and the twice transformed photo to be similar to one another. Thus, we can calculate the cycle consistency loss be finding the average of their difference.

```
with strategy.scope():
    def calc_cycle_loss(real_image, cycled_image, LAMBDA):
        loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))

        return LAMBDA * loss1
```
The identity loss compares the image with its generator (i.e. photo with photo generator). If given a photo as input, we want it to generate the same image as the image was originally a photo. The identity loss compares the input with the output of the generator.

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
    ax[i, 0].set_title("Input Photo")
    ax[i, 1].set_title("Monet-esque")
    ax[i, 0].axis("off")
    ax[i, 1].axis("off")
plt.show()
```
![monetesque.jpg](/images/monet/monet7.jpg)

<a id="ch90"></a>
# References
I would like to express gratitude for the following resources, and thank developers for the inspiration:

* [Practical Introduction to 10 Regression Algorithm](https://www.kaggle.com/code/faressayah/practical-introduction-to-10-regression-algorithm/) - Kaggle notebook going over different regression techniques.




