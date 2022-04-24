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
1. [Chapter 6 - Step 4: Explanatory Data Analysis (EDA)](#ch6)
1. [Chapter 7 - Step 5: Data Modelling](#ch7)
1. [Chapter 8 - Evaluate Model Performance](#ch8)
1. [Chapter 9 - Tune Model with Hyper-Parameters](#ch9)
1. [Chapter 10 - Tune Model with Feature Selection](#ch10)
1. [Chapter 11 - Step 6: Validate Model](#ch11)
1. [Chapter 12 - Step 7: Optimize Model](#ch12)
1. [References](#ch90)


<a id="ch1"></a>
# Project Overview
I have always loved painting and data science. I wanted to take on this project to connect both passions of mine. Painters, such as Claude Monet have unique brush strokes and color choices. This project revolves around whether I can use generative adversarial networks (GANs) to bring Monet's style to the existing photos and recreating that style from stratch.

If those photos are created successfully, the classifier will approve and I call myself a junior Monet. I trust the computer vision's recent advancements can handle this, and let us see how well I can Monet-ize it? :)

**A GAN** consists of at least two neural networks: a generator model and a discriminator model. The generator is a neural network that creates the images. For this project, I generated images in the style of Monet. This generator is trained using a discriminator. The two models will work against each other, with the generator trying to convince the discriminator, and the discriminator trying to accurately classify the real vs. generated images. I have built a GAN that generates 7,000 to 10,000 Monet-style images.

Let's take a look at the steps:  

<a id="ch2"></a>
# Data Science Steps
1. **Problem Definition:** Finding whether the image that is created is real or generated through classification. 
2. **Data Gathering:** I used the Monet TFRecord dataset as well as the Photo TFRecord dataset, I got access to them through the KaggleDatasets() app. 
3. **Data Preperation:** I prepped the data by using scaling and normalization methods.
4. **EDA (Explanatory Data Analysis):** It is essential to use descriptive and graphical statistics to look for patterns, correlations and comparisons in the dataset. In this step I mostly used visualization techniques to analyze the data. 
5. **Data Modelling:** In this project, I built a generator to generate Monet-like photos, a discriminator to classify whether the image is real or fake, CycleGAN architecture to train the model.
6. **Validate Model:** After training the model, I worked on validating it to see the performance of the model I have built.
7. **Optimize Model:** Used loss function to adjust the weights to optimize the model. 

<a id="ch3"></a>
# Step 1: Problem Definition
Goal is to generate Monet like photos either from stratch or from existing photos, and then classify them whether they are real or fake.

**Project Summary from Kaggle:**
We recognize the works of artists through their unique style, such as color choices or brush strokes. The “je ne sais quoi” of artists like Claude Monet can now be imitated with algorithms thanks to generative adversarial networks (GANs). In this getting started competition, you will bring that style to your photos or recreate the style from scratch!

Computer vision has advanced tremendously in recent years and GANs are now capable of mimicking objects in a very convincing way. But creating museum-worthy masterpieces is thought of to be, well, more art than science. So can (data) science, in the form of GANs, trick classifiers into believing you’ve created a true Monet? That’s the challenge you’ll take on!

The Challenge:
A GAN consists of at least two neural networks: a generator model and a discriminator model. The generator is a neural network that creates the images. For our competition, you should generate images in the style of Monet. This generator is trained using a discriminator.

The two models will work against each other, with the generator trying to trick the discriminator, and the discriminator trying to accurately classify the real vs. generated images.

Your task is to build a GAN that generates 7,000 to 10,000 Monet-style images.



<a id="ch4"></a>
# Step 2: Data Gathering

Dataset can be found at the Kaggle's mainpage for this project: [Kaggle: I’m Something of a Painter Myself](https://www.kaggle.com/competitions/gan-getting-started/data) or using the Kaggle app in Python. I went with the second option. 

<a id="ch5"></a>
# Step 3: Data Preperation
The data is pre-processed already coming from Kaggle so I just focused on scaling/normalizing the data further. All the images were already sized to 256x256. I also scaled the images to a [-1, 1] scale. Because we are building a generative model, we don't need the labels or the image id in this project. 

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
-------------------------

**This is the input data from Kaggle :**  ['monet_jpg', 'monet_tfrec', 'photo_jpg', 'photo_tfrec']

## 3.2 Pre-view of the Data


The dataset contained four directories: monet_tfrec, photo_tfrec, monet_jpg, and photo_jpg. The monet_tfrec and monet_jpg directories contained the same painting images, and the photo_tfrec and photo_jpg directories contained the same photos. I used the TFRecords as per the Kaggle's recommendation.

The monet directories contained Monet paintings. I used these images to train my model.

The photo directories contained photos. I added Monet's styling to these images via GAN architectures. CycleGAN dataset contains other artists styles as well. 

**Files**
monet_jpg - 300 Monet paintings sized 256x256 in JPEG format
monet_tfrec - 300 Monet paintings sized 256x256 in TFRecord format
photo_jpg - 7028 photos sized 256x256 in JPEG format
photo_tfrec - 7028 photos sized 256x256 in TFRecord format

![photo_and_monet_example.jpg](/images/monet/monet1.jpg)

<a id="ch5"></a>
## 3.3 Data Pre-processing: 

All the images for the competition were already sized to 256x256. As these images are RGB images, I set the channel to 3. Also, I needed to scale the images to a [-1, 1]. Due to the nature of the generative model, I don't need the labels or the image id so I'll only return the image from the TFRecord.

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

<a id="ch6"></a>
# Step 4: Explanatory Data Analysis (EDA)
Exploration is key after cleaning and organizing the dataset. I worked on EDA to visualize the properties and stats of the data I am working with.

Visualizing the quantitative data on graph:

![Titanic_Project_28_1.png](/images/titanic/titanic3.png)

Looking at individual features by survival:

![Titanic_Project_29_1.png](/images/titanic/titanic4.png)

I then compared class and a 2nd feature:

![Titanic_Project_30_1.png](/images/titanic/titanic5.png)

Followed by comparing sex and a 2nd feature:

![Titanic_Project_31_1.png](/images/titanic/titanic6.png)

Family size factor with sex & survival comparison and class factor with sex & survival comparison:

![Titanic_Project_32_1.png](/images/titanic/titanic7.png)

Embark data visualization with class, sex, and survival:

![Titanic_Project_33_1.png](/images/titanic/titanic8.png)

Distributions of age of passengers who survived or did not survive:

![Titanic_Project_34_1.png](/images/titanic/titanic9.png)

Histogram comparison of sex, class, and age by survival:

![Titanic_Project_35_1.png](/images/titanic/titanic10.png)

Pairplot to see the entire dataset:

![Titanic_Project_36_1.png](/images/titanic/titanic11.png)

Heatmap of the entire dataset:

![Titanic_Project_37_0.png](/images/titanic/titanic12.png)

<a id="ch7"></a>
# Step 5: Data Modelling

I will use supervised learning classification algorithm for predicting the binary ourcome (survived or not). Here are some of the available models I considered using: 

**Machine Learning Classification Algorithms:**
* [Ensemble Methods](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.ensemble)
* [Generalized Linear Models (GLM)](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model)
* [Naive Bayes](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.naive_bayes)
* [Nearest Neighbors](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.neighbors)
* [Support Vector Machines (SVM)](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.svm)
* [Decision Trees](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.tree)
* [Discriminant Analysis](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.discriminant_analysis)


### Which Machine Learning Algorithm (MLA) to choose ?
In order to define that I worked on some performance analysis for different models:

![compare_mla.jpg](/images/titanic/titanic13.jpg)

Then let's see the barplot:

![Titanic_Project_39_1.png](/images/titanic/titanic14.png)

<a id="ch8"></a>
## 5.1 Evaluate Model Performance
After some data pre-processing, analysis, and machine learning algorithms (MLA), I was able to predict passenger survival with ~82% accuracy. can I do better?


### Somethings to consider: ###
Our accuracy is increasing, but can we do better? I looked more correlations to improve the data.


This is the result of the model with improvements:


![handmade_model_score.jpg](/images/titanic/titanic15.jpg)

The confusion matrix without normalization:

![Titanic_Project_44_1.png](/images/titanic/titanic16.png)

Confusion matrix with normalization:

![Titanic_Project_44_2.png](/images/titanic/titanic17.png)

## 5.11 Model Performance with Cross-Validation (CV)
In this section, I worked on cross valdiation (CV). By using CV I was autamatically able to split and score the model multiple times, to can get an idea of how well it will perform on unseen data.

<a id="ch9"></a>
# 5.12 Tune Model with Hyper-Parameters
I worked on hyper-parameter optimization to see how various hyper-parameter settings will change the model accuracy. 

Decision trees are simple to understand usually. They can also be visualized. Data prep is quite easy compared to other methods. They can handle both numeric and categorical data. We can validate a model using tests.

However, decision trees do not generalize data well, they do have tendency to memorize (overfitting). Pruning can be used to overcome this issue. Small variations may impact the decision trees hugely. They can be biased if some classes dominate.


DT before any after optimization:

![dt_parameters.jpg](/images/titanic/titanic18.jpg)

<a id="ch10"></a>
## 5.13 Tune Model with Feature Selection
Recursive feature elimination (RFE) with cross validation (CV) is used for feature selection:

![feature_elimination.jpg](/images/titanic/titanic19.jpg)

The graph visualization of the tree:

![dt_graph.jpg](/images/titanic/titanic20.jpg)

<a id="ch11"></a>
# Step 6: Validate Model
The next step is to validate the data.

Comparison of algorithm predictions with each other, where 1 = similar and 0 = opposite in a heatmap:

![Titanic_Project_54_0.png](/images/titanic/titanic21.png)

I worked on using more than one model instead of picking one. This gave an opportunity to create a supermodel. I removed the models 
who are exactly correlated to another model (1) and the models with no predict_proba attribute are also removed. 

I then worked on hard vote or majority rules and soft vote or weighted probabilities. I tuned each estimator before creating a super model


<a id="ch12"></a>
# Step 7: Optimize Model
## Conclusion

Model provides ~0.78 submission accuracy on the unseen data which was achieved with the simple decision tree. Using the same dataset and different implementation of a decision tree with a super model (adaboost, random forest, gradient boost, xgboost, etc.) with tuning does not exceed the ~0.78 submission accuracy. Conclusion was the simple decision tree algorithm had the best default submission score and with tuning, I still achieved the same best accuracy score.
 
-  The train dataset has a different distribution than the test/validation dataset and population. This created wide margins between the cross validation (CV) accuracy score and Kaggle submission accuracy score.
- Given the same dataset, decision tree based algorithms, seemed to converge on the same accuracy score after proper tuning.
-  Despite tuning, no machine learning algorithm, exceeded the homemade algorithm. The author will theorize, that for small datasets, a manmade algorithm is the bar to beat. 

Next steps will include further preprocessing and feature engineering to improve the CV score and Kaggle score as well as the overall accuracy.



<a id="ch90"></a>
# References
I would like to express gratitude for the following resources, and thank developers for the inspiration:

* [A Data Science Framework: To Achieve 99% Accuracy by LD Freeman](https://www.kaggle.com/ldfreeman3/a-data-science-framework-to-achieve-99-accuracy) - Indepth dive to data science steps.

* [Introduction to Machine Learning with Python: A Guide for Data Scientists by Andreas Müller and Sarah Guido](https://www.amazon.com/gp/product/1449369413/ref=as_li_tl?ie=UTF8&tag=kaggle-20&camp=1789&creative=9325&linkCode=as2&creativeASIN=1449369413&linkId=740510c3199892cca1632fe738fb8d08) - Machine Learning 101 written by a core developer of sklearn
* [Visualize This: The Flowing Data Guide to Design, Visualization, and Statistics by Nathan Yau](https://www.amazon.com/gp/product/0470944889/ref=as_li_tl?ie=UTF8&tag=kaggle-20&camp=1789&creative=9325&linkCode=as2&creativeASIN=0470944889&linkId=f797da48813ed5cfc762ce5df8ef957f) - Learn the art and science of data visualization
* [Machine Learning for Dummies by John Mueller and Luca Massaron ](https://www.amazon.com/gp/product/1119245516/ref=as_li_tl?ie=UTF8&tag=kaggle-20&camp=1789&creative=9325&linkCode=as2&creativeASIN=1119245516&linkId=5b4ac9a6fd1da198d82f9ca841d1af9f) - Easy to understand for a beginner book, but detailed to actually learn the fundamentals of the topic




