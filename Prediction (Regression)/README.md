![Let's look at the home prices!](/images/house/house0.jpg)

In this project, I will use regression analysis to calculate home prices. This project uses Kaggle datasets and draws inspiration from public notebooks.

# Table of Contents
1. [Chapter 1 - Project Overview](#ch1)
1. [Chapter 2 - Data Science Steps](#ch2)
1. [Chapter 3 - Step 1: Problem Definition](#ch3)
1. [Chapter 4 - Step 2: Data Gathering](#ch4)
1. [Chapter 5 - Step 3: EDA and Data Preparation](#ch5)
1. [Chapter 6 - Step 4: Build the Models](#ch6)
1. [Chapter 7 - Step 5: Model Comparison](#ch7)
1. [Chapter 7 - Step 6: Summary](#ch8)


1. [References](#ch90)


<a id="ch1"></a>
# Project Overview
Housing has been very popular in the past few years. Also, one of my personal interests is participating in projects that predict housing prices. I saw this project on Kaggle where there are explanatory variables for most aspects of residential homes in Ames, Iowa. We are being asked to estimate the final prices of each home.

Let's take a look at how to do this:  

<a id="ch2"></a>
# Data Science Steps
1. **Problem Definition:** How much will the homes cost? Since house prices are a continuous variable, this is a regression problem.
2. **Data Gathering:** I used the USA_Housing dataset, which I got access to through the [Kaggle: House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data).  
3. **Data Preparation:** I prepared the data by using scaling methods..
4. **EDA (Exploratory Data Analysis):** It is important to use descriptive and graphical statistics to look for patterns, correlations, and comparisons in the dataset. In this step, I used heatmaps and correlation matrices to analyze the data. 
5. **Data Modelling:** In this project, I used different linear regression methods including Linear Regression, Robust Regression, Ridge Regression, LASSO Regression,  Polynomial Regression, Stochastic Gradient Descent, Artficial Neural Network,  Random Forest Regressor and Support Vector Machine.
6. **Validate Model:** After training the model, I used cross-validation techniques to validate the model.
7. **Optimize Model:** In this particular project, I didn't focus on optimizing the model, but used the models from sklearn with constant parameters. Parameter optimization will be a future improvement that I will be working on. 

<a id="ch3"></a>
# Step 1: Problem Definition
The goal of this study is to predict the home prices using known variables.

**Project Summary from Kaggle:**
What is your dream house like? The dataset from this playground competition proves that more factors affect price negotiations than the number of bedrooms or a white-picket fence.

This competition challenges participants to predict the final price of each home in Ames, Iowa using 79 explanatory variables.

Practice Skills: 

- Creative feature engineering 
- Advanced regression techniques like random forest and gradient boosting

<a id="ch4"></a>
# Step 2: Data Gathering

The dataset for this project can be found on the Kaggle website: [House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) or using the Kaggle app in Python. 

**This is the input data from Kaggle :**  ['USA_Housing.csv']

I used the USA_Housing dataset. The data contains the following columns:

- <b> Avg. Area Income </b>: Avg. Income of residents of the city house is located in.
- <b> Avg. Area House Age </b>: Avg. Age of Houses in the same city.
- <b> Avg. Area Number of Rooms </b>: Avg. Number of Rooms for Houses in the same city.
- <b> Avg. Area Number of Bedrooms </b>: Avg. Number of Bedrooms for Houses in the same city.
- <b> Area Population </b>: Population of the city in which the house is located.
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
USAhousing.head(10)
```

![data_head.jpg](/images/house/house1.jpg)

```
USAhousing.info()
```

![data_head.jpg](/images/house/house2.jpg)

The dataset does not contain any null values.

<a id="ch5"></a>
## 3.3 Exploratory Data Analysis (EDA): 

I examined the data in more detail using pair plots and heatmaps.

![pair_plots.jpg](/images/house/house3.jpg)

![pair_plots.jpg](/images/house/house4.jpg)

## 3.3 Data Preperation: 

The Kaggle dataset arrived in a cleaned-up form. In this section, I have divided the data into pieces that I can work on separately. First, the data is split into an X array containing the training functions, and an y array with the target Price variable. In this case, I got rid of the address column as it contained only text.

```
X = USAhousing[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms', 'Avg. Area Number of Bedrooms', 'Area Population']]
y = USAhousing['Price']
```

I split the data into test and train sets. The train set will be used to train the model, while the test set will be used to test the model's performance.

```
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

Standardizing the data is a very important step before making the model. I standardized the data using the StandardScaler() function. The model performance functions were defined to show how the model performs.

<a id="ch6"></a>
# Step 4: Build the Models

## 4.1 Linear Regression

- Linear Assumption: Linear regression assumes that the relationship between input and output is linear. In most cases, raw data may not be linear in real life. There are methods to make the relationship linear (e.g. "Linearize" an exponential-looking graph with log function).
- Remove Noise: If possible, outliers should be removed from the data before it is used in analysis. This will help to make the data more reliable and accurate. 
- Remove Collinearity: Linear regression can be prone to overfitting the data when the input variables are highly correlated. This is mainly why I looked at the pairplots to see if I could remove something I didn't see. 
- Normal Distribution: If the input and output variables have a normal distribution, linear regression is more accurate in predicting outcomes. There are some ways to make the distribution of the variables more normal.
- Rescaling : Linear regression often produces more reliable predictions if we scale the input variables using standardization or normalization.

The sklearn library makes it easy to apply the models: 

```
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression(normalize=True)
lin_reg.fit(X_train,y_train)
```

Then, I visualized the true values and predicted values for the data.

![pair_plots.jpg](/images/house/house5.jpg)

We can take a look at the residual histogram:

![pair_plots.jpg](/images/house/house6.jpg)

Now that we've covered the basics, let's get to the evaluation. This article is helpful: [what metrics to use when evaluating the regression models](https://towardsdatascience.com/what-are-the-best-metrics-to-evaluate-your-regression-model-418ca481755b)

There are three main metrics for evaluating regression models:
1. R Square/Adjusted R Square : R Square measures how well the model explains the variation in the dependent variable.
2. Mean Square Error(MSE)/Root Mean Square Error(RMSE): While R Square is a measure of how well the model fits dependent variables, Mean Square Error is an absolute measure of the goodness of the fit.
3. Mean Absolute Error(MAE): Mean absolute error (MAE) is similar to root mean square error (MSE). However, MAE is taking the sum of the absolute value of the error, rather than the square of the error in MSE.

Next, I printed the evaluation metrics for the linear regression model.

![linear_regression.jpg](/images/house/house7.jpg)

## 4.2 Robust Regression

As stated at the [Robust Regression resource by UCLA](https://stats.oarc.ucla.edu/r/dae/robust-regression/): Robust regression is an alternative to least squares regression when the data is contaminated by outliers or influential observations, and can also be used to detect influential observations. A robust regression technique is often used when the data contain outliers.

I used the Random sample consensus (RANSAC) model to generate a prediction. The estimation method is an iterative process that takes into account outliers without any significant impact.

Here are the results for the Robust Regression model:

![robust_regression.jpg](/images/house/house8.jpg)

## 4.3 Ridge Regression

There is no shortage of excellent resources on this topic, one great resource can be found here: [Ridge Regression](https://www.mygreatlearning.com/blog/what-is-ridge-regression/). As is explained in the referenced resource, Ridge regression is a model tuning method that is useful for analysing data that is plagued by multicollinearity. This method produces high-quality results with L2 regularization. When multicollinearity is a problem, least-squares methods are unbiased, and variances are large, which means predicted values will be far from the actual values.


In summary this is what Ridge Regression does:

- It reduces the size of the parameters. Therefore, it is used to reduce the likelihood of multicollinearity.
- The model is simplified by coefficient shrinkage.

Here are the results for the Ridge Regression model:

![ridge_regression.jpg](/images/house/house9.jpg)

## 4.4 Lasso Regression

The same resource I extensively used during the Ridge Regression part can be used for the [Lasso Regression](https://www.mygreatlearning.com/blog/understanding-of-lasso-regression/#lassoregression).  Lasso regression is a technique that helps to regularize data. The regression method is more accurate when used in conjunction with other methods. This model uses shrinkage, which is where data values are shrunk towards a central point as the mean. The lasso procedure encourages simple, sparse models which are easier to understand and work with. Many models have fewer parameters than those with more parameters. This regression method is well-suited for models with high levels of multicollinearity or when you want to automate certain parts of model selection, like variable selection/parameter elimination.

Lasso regression is a technique that uses L1 regularization. Feature selection is a process of selecting the features that are most important for a given application, that process is perfomed automatically here. Thus it is used when we have more number of features. 

Here are the results of the Lasso Regression model:

![lasso_regression.jpg](/images/house/house10.jpg)

## 4.5 Polynomial Regression

[Polynomial Regression](https://www.analyticsvidhya.com/blog/2021/07/all-you-need-to-know-about-polynomial-regression/) is a form of linear regression where only because of the non-linear relationship between the dependent and independent variables we add some polynomial terms to the linear regression to turn it into a polynomial regression.

We have X as data that is independent of Y, and Y as data that is dependent on X. Before feeding data to a mode in preprocessing stage, we convert the input variables into polynomial terms using some degree. Simple linear regression can be extended by constructing polynomial features from the coefficients.

Here are the results of the Polynomial Regression model:

![polynomial_regression.jpg](/images/house/house11.jpg)

## 4.6 Stochastic Gradient Descent
 Gradient descent is a technique for adjusting the parameters of a model or algorithm in order to reduce the cost function. Gradient descent searches for the smallest error in a function by increasing the value of the parameters in the direction of the steepest descent. When the gradient reaches zero, it reaches the minimum value. There is a good explanation for this phenomena with analogies from the real world: [All you need to know about Gradient Descent](https://medium.com/analytics-vidhya/all-you-need-to-know-about-gradient-descent-f0178c19131d).

Here are the results of the Stochastic Gradient Descent model:

![stochastic_gradient.jpg](/images/house/house12.jpg)

## 4.7 Neural Networks

Artificial neural networks are made up of a layer of nodes, each of which has an input layer, one or more hidden layers, and an output layer. Each node has an associated weight and threshold. When a node is activated, it sends a signal to other nodes that are connected to it. The signal has a weight and threshold associated with it, which determines how strongly the node is activated.If the output of any node is above a certain threshold value, that node is activated and sends data to the next layer of the network. Otherwise, no data is passed on to the next layer of the network. For more information on that: [Neural Networks](https://www.ibm.com/cloud/learn/neural-networks). 

Here are the results of the Neural Networks model:

![neural_net1.jpg](/images/house/house13.jpg)
![neural_net2.jpg](/images/house/house14.jpg)
![neural_net3.jpg](/images/house/house15.jpg)

## 4.8 Random Forest Regressor

Random Forest Regression is a supervised learning algorithm that uses ensemble learning method for regression. The ensemble learning method is a technique that uses predictions from multiple machine learning algorithms to make a more accurate prediction than a single model could. More on that can be found at [Random Forest Regressor](https://levelup.gitconnected.com/random-forest-regression-209c0f354c84).

Here are the results of the Random Forest Regressor model:

![neural_net1.jpg](/images/house/house16.jpg)

## 4.9 Support Vector Machine 

Support vector machines are commonly used for classification and regression methods. I used this tool for regression in this particular project. The Support Vector Regression (SVR) uses the same principles as the Support Vector Machine (SVM) for classification, with a few minor differences. A margin of tolerance (epsilon) is set to approximate the effectiveness of a SVM. The goal is to minimize error by individually tailoring the hyperplane which maximizes the margin. For more information on that topic, you can find it at the following address: [Support Vector Machine ](https://www.saedsayad.com/support_vector_machine_reg.htm).

Here are the results of the Support Vector Machine model:

![neural_net1.jpg](/images/house/house17.jpg)

<a id="ch7"></a>
# Step 5: Model Comparison 

Let's see all of the models together so we can compare them: 

![neural_net1.jpg](/images/house/house18.jpg)


<a id="ch8"></a>
# Step 6: Summary

In this project, I learned about different regression models and how to evaluate them using different metrics. It was a great way to explore regression and experiment with different models. As a next step, I plan to focus on optimizing hyperparameters and getting a better understanding of the model's parameters. 

<a id="ch90"></a>
# References
Thank you for the following resources and developers for the inspiration:

* [Practical Introduction to 10 Regression Algorithm](https://www.kaggle.com/code/faressayah/practical-introduction-to-10-regression-algorithm/) - Kaggle notebook going over different regression techniques.
* [House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
* [what metrics to use when evaluating the regression models](https://towardsdatascience.com/what-are-the-best-metrics-to-evaluate-your-regression-model-418ca481755b)
* [Robust regression resource by UCLA](https://stats.oarc.ucla.edu/r/dae/robust-regression/)
* [Ridge Regression](https://www.mygreatlearning.com/blog/what-is-ridge-regression/)
* [Lasso Regression](https://www.mygreatlearning.com/blog/understanding-of-lasso-regression/#lassoregression)
* [Polynomial Regression](https://www.analyticsvidhya.com/blog/2021/07/all-you-need-to-know-about-polynomial-regression/)
* [All you need to know about Gradient Descent](https://medium.com/analytics-vidhya/all-you-need-to-know-about-gradient-descent-f0178c19131d)
* [Neural Networks](https://www.ibm.com/cloud/learn/neural-networks)
* [Random Forest Regressor](https://levelup.gitconnected.com/random-forest-regression-209c0f354c84)
* [Support Vector Machine ](https://www.saedsayad.com/support_vector_machine_reg.htm)