![I am something of a painter myself]

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
In this project,I picked a very famous example for classificaion. It is 0 or 1. It either occurred or did not occur. For example, cancer positive or not, manufacturing part pass or fail, it will rain tomorrow or won't etc. 

In this case, I am focusing on passingers survived the disaster or not. I used Kaggle's Getting Started Competition, Titanic: Machine Learning from Disaster. I followed a some steps such as focusing on problem definition, gathering the data, preparing the data, EDA (explanatory data analysis), coming up with the data model, validating the model and optimizing the model further.




Let's take a look at the steps:
   


<a id="ch2"></a>
# Data Science Steps
1. **Problem Definition:** Finding "what sorts of people were more likely to survive?” using passenger data (ie name, age, gender, socio-economic class, etc).
2. **Data Gathering:** Kaggle provided the input data on their website. That is how I got access to them.
3. **Data Preperation:** I prepped the data by analyzing missing, or outlier data points.
4. **EDA (Explanatory Data Analysis):** Garbage-in, garbage-out (GIGO). Therefore, it is essential to use descriptive and graphical statistics to look for patterns, correlations and comparisons in the dataset. In this step I made sure to make sense of the data. 
5. **Data Modelling:** It is very important to know when to select which model. If we select the wrong model for a particular usecase, all the other steps become meaningless. 
6. **Validate Model:** After training the model, I worked on validating it to see the performance and the overfitting/underfitting issues.
7. **Optimize Model:** Using techniques like hyperparameter optimization, I worked on making the model better.  

<a id="ch3"></a>
# Step 1: Problem Definition
Goal is to predict the survival outcome of passengers on the Titanic.

**Project Summary from Kaggle:**
The sinking of the Titanic is one of the most infamous shipwrecks in history.

On April 15, 1912, during her maiden voyage, the widely considered “unsinkable” RMS Titanic sank after colliding with an iceberg. Unfortunately, there weren’t enough lifeboats for everyone onboard, resulting in the death of 1502 out of 2224 passengers and crew.

While there was some element of luck involved in surviving, it seems some groups of people were more likely to survive than others.

In this challenge, we ask you to build a predictive model that answers the question: “what sorts of people were more likely to survive?” using passenger data (ie name, age, gender, socio-economic class, etc).



<a id="ch4"></a>
# Step 2: Data Gathering

Dataset can be found at the Kaggle's mainpage for this project: [Kaggle's Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic/data)

<a id="ch5"></a>
# Step 3: Data Preperation
The data is pre-processed already coming from Kaggle so I just focused on cleaning the data further. 

## 3.1 Import Libraries


**These were the versions of the libraries I used**

Python version: 3.9.7 (default, Sep 16 2021, 16:59:28) [MSC v.1916 64 bit (AMD64)]
pandas version: 1.3.4
matplotlib version: 3.4.3
NumPy version: 1.20.3
SciPy version: 1.7.1
IPython version: 8.0.1
scikit-learn version: 0.24.2
-------------------------

**This is the input data from Kaggle :**  ['gender_submission.csv', 'test.csv', 'train.csv']

## 3.2 Pre-view of the Data


1. The *Survived* variable is the outcome or dependent variable. It is a binary nominal datatype of 1 for survived and 0 for did not survive. All other variables are potential predictor or independent variables. 
2. The *PassengerID* and *Ticket* variables are assumed to be random unique identifiers, that have no impact on the outcome variable. Thus, they will be excluded from analysis.
3. The *Pclass* variable is an ordinal datatype for the ticket class, a proxy for socio-economic status (SES), representing 1 = upper class, 2 = middle class, and 3 = lower class.
4. The *Name* variable is a nominal datatype. It could be used in feature engineering to derive the gender from title, family size from surname, and SES from titles like doctor or master. Since these variables already exist, we'll make use of it to see if title, like master, makes a difference.
5. The *Sex* and *Embarked* variables are a nominal datatype. They will be converted to dummy variables for mathematical calculations.
6. The *Age* and *Fare* variable are continuous quantitative datatypes.
7. The *SibSp* represents number of related siblings/spouse aboard and *Parch* represents number of related parents/children aboard. Both are discrete quantitative datatypes. This can be used for feature engineering to create a family size and is alone variable.
8. The *Cabin* variable is a nominal datatype that can be used in feature engineering for approximate position on ship when the incident occurred and SES from deck levels. However, since there are many null values, it does not add value and thus is excluded from analysis.

![pre-view_dataframe.jpg](/images/titanic/titanic1.jpg)

![dataframe.jpg](/images/titanic/titanic2.jpg)

<a id="ch5"></a>
## 3.3 Data Pre-processing: 
In this stage, I cleaned the data by analyzing aberrant values and outliers, filled in missing data where appropriate, worked on feature engineering, and performed data conversion (i.e. convert objects to category using Label Encoder)

I splitted the data in 75/25 format, 75 being the training and 25 being the testing. I paid attention to this split as to not overfit or underfit the model.

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




