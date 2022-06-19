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
1. [Chapter 7 - Step 6: Summary](#ch8)


1. [References](#ch90)


<a id="ch1"></a>
# Project Overview
Housing has been super hot, especially for the last couple of years. Also, It has been one of my personal interests to work on a project to predict housing prices. I come across this project in Kaggle, in this project, there are explanatory variables describing most aspects of residential homes in Ames, Iowa. The task is to estimate the final price of each home. 

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
USAhousing.head(10)
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

![robust_regression.jpg](/images/house/house8.jpg)

## 4.3 Ridge Regression

There is a great resource on [Ridge Regression](https://www.mygreatlearning.com/blog/what-is-ridge-regression/). As they explain in the above resource, Ridge regression is a model tuning method that is used to analyse any data that suffers from multicollinearity. This method performs L2 regularization. When the issue of multicollinearity occurs, least-squares are unbiased, and variances are large, this results in predicted values being far away from the actual values. 


In summary this is what Ridge Regression does:

- It shrinks the parameters. Therefore, it is used to prevent multicollinearity
- It reduces the model complexity by coefficient shrinkage

Here are the results for my Ridge Reegression model:

![ridge_regression.jpg](/images/house/house9.jpg)

## 4.4 Lasso Regression

Same resource I highly benefited from above during the Ridge Regression part covers the [Lasso Regression](https://www.mygreatlearning.com/blog/understanding-of-lasso-regression/#lassoregression). As they state on the website: Lasso regression is a regularization technique. It is used over regression methods for a more accurate prediction. This model uses shrinkage. Shrinkage is where data values are shrunk towards a central point as the mean. The lasso procedure encourages simple, sparse models (i.e. models with fewer parameters). This particular type of regression is well-suited for models showing high levels of multicollinearity or when we want to automate certain parts of model selection, like variable selection/parameter elimination.

Lasso Regression uses L1 regularization technique. It is used when we have more number of features because it automatically performs feature selection.

Here are the results of the Lasso Regression model:

![lasso_regression.jpg](/images/house/house10.jpg)

## 4.5 Polynomial Regression

[Polynomial Regression](https://www.analyticsvidhya.com/blog/2021/07/all-you-need-to-know-about-polynomial-regression/) is a form of Linear regression where only due to the non-linear relationship between dependent and independent variables we add some polynomial terms to linear regression to convert it into Polynomial regression.

Suppose we have X as Independent data and Y as dependent data. Before feeding data to a mode in preprocessing stage we convert the input variables into polynomial terms using some degree. A simple linear regression can be extended by constructing polynomial features from the coefficients.

Here are the results of the Polynomial Regression model:

![polynomial_regression.jpg](/images/house/house11.jpg)

## 4.6 Stochastic Gradient Descent
 The general idea of Gradient Descent is to change the parameters iteratively in order to minimize a cost function. Gradient Descent measures the local gradient of the error function with regards to the parameters vector, and it goes in the direction of descending gradient. Once the gradient is zero, we have reached a minimum. A good explanation with analogies from real wold can be found at [All you need to know about Gradient Descent
](https://medium.com/analytics-vidhya/all-you-need-to-know-about-gradient-descent-f0178c19131d).

Here are the results of the Stochastic Gradient Descent model:

![stochastic_gradient.jpg](/images/house/house12.jpg)

## 4.7 Neural Networks

Artificial neural networks (ANNs) are comprised of a node layers, containing an input layer, one or more hidden layers, and an output layer. Each node, or artificial neuron, connects to another and has an associated weight and threshold. If the output of any individual node is above the specified threshold value, that node is activated, sending data to the next layer of the network. Otherwise, no data is passed along to the next layer of the network. More on that can be found at [Neural Networks](https://www.ibm.com/cloud/learn/neural-networks). 

Here are the results of the Neural Networks model:

![neural_net1.jpg](/images/house/house13.jpg)
![neural_net2.jpg](/images/house/house14.jpg)
![neural_net3.jpg](/images/house/house15.jpg)

## 4.8 Random Forest Regressor

Random Forest Regression is a supervised learning algorithm that uses ensemble learning method for regression. Ensemble learning method is a technique that combines predictions from multiple machine learning algorithms to make a more accurate prediction than a single model. More on that can be found at [Random Forest Regressor](https://levelup.gitconnected.com/random-forest-regression-209c0f354c84).

Here are the results of the Random Forest Regressor model:

![neural_net1.jpg](/images/house/house16.jpg)

## 4.9 Support Vector Machine 
![neural_net1.jpg](/images/house/house17.jpg)

Support Vector Machine be used for classification and regression methods. In this particular project, I used it for rgression. The Support Vector Regression (SVR) uses the same principles as the SVM for classification, with only a few minor differences. Because output is a real number, it becomes very difficult to predict the information at hand, which has infinite possibilities. In the case of regression, a margin of tolerance (epsilon) is set in approximation to the SVM. The main idea is to minimize error, individualizing the hyperplane which maximizes the margin. More on that can be found at [Support Vector Machine ](https://www.saedsayad.com/support_vector_machine_reg.htm).

<a id="ch7"></a>
# Step 5: Model Comparison 

Let's see all of the models in one place for comparison: 

![neural_net1.jpg](/images/house/house18.jpg)


<a id="ch8"></a>
# Step 6: Summary

In this project I went over different regression models and their evaluation metrics. It was a great way to go deeper into regression and experimenting with differnt models. As a next step, I plan to focus on introducing hyperparameter optimization and diving deeper into the model parameters. 

<a id="ch90"></a>
# References
I would like to express gratitude for the following resources, and thank developers for the inspiration:

* [Practical Introduction to 10 Regression Algorithm](https://www.kaggle.com/code/faressayah/practical-introduction-to-10-regression-algorithm/) - Kaggle notebook going over different regression techniques.




