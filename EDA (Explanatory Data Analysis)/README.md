![Exploratory data analysis for breast cancer prediction](/images/breast_cancer/breast_cancer0.jpg)

This project focuses on exploratory data analysis with a simple step-by-step explanation and uses Kaggle datasets and gets inspiration from public notebooks.

# Table of Contents
1. [Chapter 1 - Project Overview](#ch1)
1. [Chapter 2 - EDA Definition and Steps](#ch2)
1. [Chapter 3 - Step 1: Data Gathering](#ch3)
1. [Chapter 4 - Step 2: Head and describe](#ch4)
1. [Chapter 5 - Step 3: Target distribution](#ch5)
1. [Chapter 6 - Step 4: Features distribution](#ch6)
1. [Chapter 7 - Step 5: Correlation matrix](#ch7)
1. [Chapter 8 - Step 6: Positive correlated features](#ch8)
1. [Chapter 9 - Step 7: Uncorrelated features](#ch9)
1. [Chapter 10 - Step 8: Negative correlated features](#ch10)

1. [References](#ch90)


<a id="ch1"></a>
# Project Overview

# **Project Summary from Kaggle:**

Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image. n the 3-dimensional space is that described in: [K. P. Bennett and O. L. Mangasarian: "Robust Linear Programming Discrimination of Two Linearly Inseparable Sets", Optimization Methods and Software 1, 1992, 23-34].

This database is also available through the UW CS ftp server:
- ftp ftp.cs.wisc.edu
- cd math-prog/cpo-dataset/machine-learn/WDBC/

[Also can be found on UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29)

# Attribute information:

- ID number
- Diagnosis (M = Malignant(Cancerous) or B = Benign(Not Cancerous))

# Ten real-valued features are computed for each cell nucleus:

- radius (mean of distances from center to points on the perimeter)
- texture (standard deviation of gray-scale values)
- perimeter
- area
- smoothness (local variation in radius lengths)
- compactness (perimeter^2 / area - 1.0)
- concavity (severity of concave portions of the contour)
- concave points (number of concave portions of the contour)
- symmetry
- fractal dimension ("coastline approximation" - 1)

The mean, standard error and "worst" or largest (mean of the three largest values) of these features were computed for each image, resulting in 30 features. For instance, field 3 is Mean Radius, field 13 is Radius SE, field 23 is Worst Radius.

* All feature values are recoded with four significant digits

* Missing attribute values: none

* Class distribution: 357 benign, 212 malignant

Let's take a look at the steps:  

<a id="ch2"></a>
# EDA Definition and Steps

![Exploratory data analysis steps](/images/breast_cancer/breast_cancer1.jpg)

Exploratory data analysis (EDA) is used by data scientists to analyze and investigate data sets and summarize their main characteristics, often employing data visualization methods. It helps determine how best to manipulate data sources to get the answers you need, making it easier for data scientists to discover patterns, spot anomalies, test a hypothesis, or check assumptions.

EDA is primarily used to see what data can reveal beyond the formal modeling or hypothesis testing task and provides a provides a better understanding of data set variables and the relationships between them. It can also help determine if the statistical techniques you are considering for data analysis are appropriate. Originally developed by American mathematician John Tukey in the 1970s, EDA techniques continue to be a widely used method in the data discovery process today

It is a study that focuses on advanced functionalized exploratory data analysis with a simple step-by-step explanation. It involves organizing and summarizing the raw data, discovering important features and patterns in the data and any striking deviations from those patterns, and then interpreting the findings. 

Some outcomes of EDA:

- Describing the distribution of a single variable (center, spread, shape, outliers)
- Checking data (for errors or other problems)
- Checking assumptions to more complex statistical analyses
- Investigating relationships between variables

Here are the steps of EDA I will focus on in this summary:

- Head and describe
- Target distribution 
- Features distribution 
- Correlation matrix
- Positive correlated features
- Uncorrelated features
- Negative correlated features 

<a id="ch3"></a>
# Step 1: Data Gathering

Dataset can be found at the Kaggle's mainpage for this project: [Kaggle: Breast Cancer Prediction](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data) or using the Kaggle app in Python.  
```
# Read data
data = pd.read_csv('../input/data.csv')
```

I chekced the missing values: All features are complete, only 'Unnamed: 32' is completely null, probably an error in the dataset, I dropped it below:

```
# Drop unnecessary variables
data = data.drop(['Unnamed: 32','id'],axis = 1)

# Reassign target
data.diagnosis.replace(to_replace = dict(M = 1, B = 0), inplace = True)
```
<a id="ch4"></a>

# Step 2: Head and describe

```
# Head
data.head(10)
```
![Data head](/images/breast_cancer/breast_cancer2.jpg)

```
# describe
data.describe()
```
![Data describe](/images/breast_cancer/breast_cancer3.jpg)

<a id="ch5"></a>
# Step 3: Target distribution

```
# We have two datasets, which consist of malignant and benign data.
M = data[(data['diagnosis'] != 0)]
B = data[(data['diagnosis'] == 0)]
```

```
#First, we will plot the count
trace = go.Bar(x = (len(M), len(B)), y = ['malignant', 'benign'], orientation = 'h', opacity = 0.8, marker=dict(
        color=[ 'green', 'blue'],
        line=dict(color='#000000',width=1.5)))

layout = dict(title =  'Count of diagnosis variable for malignant and benign data')
                    
fig = dict(data = [trace], layout=layout)
py.iplot(fig)

#Then, we will plot the percentage
trace = go.Pie(labels = ['benign','malignant'], values = data['diagnosis'].value_counts(), 
               textfont=dict(size=15), opacity = 0.8,
               marker=dict(colors=['blue', 'green'], 
                           line=dict(color='#000000', width=1.5)))


layout = dict(title =  'Distribution of diagnosis variable for malignant and benign data')
           
fig = dict(data = [trace], layout=layout)
py.iplot(fig)

```

![Count of Diagnosis](/images/breast_cancer/breast_cancer4.jpg)
![Distribution of Diagnosis](/images/breast_cancer/breast_cancer5.jpg)

<a id="ch5"></a>
# Step 4: Features distribution 

```
def plot_distribution(data_select, size_bin) :  
    tmp1 = M[data_select]
    tmp2 = B[data_select]
    hist_data = [tmp1, tmp2]
    
    group_labels = ['malignant', 'benign']
    colors = ['#228C22', '#0000FF']

    fig = ff.create_distplot(hist_data, group_labels, colors = colors, show_hist = True, bin_size = size_bin, curve_type='kde')
    
    fig['layout'].update(title = data_select)

    py.iplot(fig, filename = 'Density plot for malignant and benign data')
```

```
#plot distribution 'mean'
plot_distribution('radius_mean', .5)
plot_distribution('texture_mean', .5)
plot_distribution('perimeter_mean', 5)
plot_distribution('area_mean', 10)
```

![Mean for Radius](/images/breast_cancer/breast_cancer6.jpg)
![Mean for Texture](/images/breast_cancer/breast_cancer7.jpg)
![Mean for Perimeter](/images/breast_cancer/breast_cancer8.jpg)
![Mean for Area](/images/breast_cancer/breast_cancer9.jpg)

```
#plot distribution 'se'
plot_distribution('radius_se', .1)
plot_distribution('texture_se', .1)
plot_distribution('perimeter_se', .5)
plot_distribution('area_se', 5)
```
![Se for Radius](/images/breast_cancer/breast_cancer10.jpg)
![Se for Texture](/images/breast_cancer/breast_cancer11.jpg)
![Se for Perimeter](/images/breast_cancer/breast_cancer12.jpg)
![Se for Area](/images/breast_cancer/breast_cancer13.jpg)

```
#plot distribution 'worst'
plot_distribution('radius_worst', .5)
plot_distribution('texture_worst', .5)
plot_distribution('perimeter_worst', 5)
plot_distribution('area_worst', 10)
```
![Worst for Radius](/images/breast_cancer/breast_cancer14.jpg)
![Worst for Texture](/images/breast_cancer/breast_cancer15.jpg)
![Worst for Perimeter](/images/breast_cancer/breast_cancer16.jpg)
![Worst for Area](/images/breast_cancer/breast_cancer17.jpg)

<a id="ch6"></a>
# Step 5: Correlation matrix
```
#correlation
correlation = data.corr()
#tick labels
matrix_cols = correlation.columns.tolist()
#convert to array
corr_array  = np.array(correlation)
```
```
#Plotting
trace = go.Heatmap(z = corr_array,
                   x = matrix_cols,
                   y = matrix_cols,
                   xgap = 2,
                   ygap = 2,
                   colorscale='Viridis',
                   colorbar   = dict() ,
                  )
layout = go.Layout(dict(title = 'Correlation Matrix for variables',
                        autosize = False,
                        height  = 720,
                        width   = 800,
                        margin  = dict(r = 0 ,l = 210,
                                       t = 25,b = 210,
                                     ),
                        yaxis   = dict(tickfont = dict(size = 9)),
                        xaxis   = dict(tickfont = dict(size = 9)),
                       )
                  )
fig = go.Figure(data = [trace],layout = layout)
py.iplot(fig)
```

![Correlation Matrix](/images/breast_cancer/breast_cancer18.jpg)

<a id="ch7"></a>
# Step 6: Positive correlated features

```
palette ={0 : 'lightblue', 1 : 'gold'}
edgecolor = 'grey'

# Plot +
fig = plt.figure(figsize=(12,12))

plt.subplot(221)
ax1 = sns.scatterplot(x = data['perimeter_mean'], y = data['radius_worst'], hue = "diagnosis",
                    data = data, palette = palette, edgecolor=edgecolor)
plt.title('perimeter mean vs radius worst')
plt.subplot(222)
ax2 = sns.scatterplot(x = data['area_mean'], y = data['radius_worst'], hue = "diagnosis",
                    data = data, palette =palette, edgecolor=edgecolor)
plt.title('area mean vs radius worst')
plt.subplot(223)
ax3 = sns.scatterplot(x = data['texture_mean'], y = data['texture_worst'], hue = "diagnosis",
                    data = data, palette =palette, edgecolor=edgecolor)
plt.title('texture mean vs texture worst')
plt.subplot(224)
ax4 = sns.scatterplot(x = data['area_worst'], y = data['radius_worst'], hue = "diagnosis",
                    data = data, palette =palette, edgecolor=edgecolor)
plt.title('area mean vs radius worst')

fig.suptitle('Positive correlated features', fontsize = 20)
plt.savefig('1')
plt.show()
```
![Positive correlated features](/images/breast_cancer/breast_cancer19.jpg)


<a id="ch8"></a>
# Step 6: Uncorrelated features

```
fig = plt.figure(figsize=(12,12))

plt.subplot(221)
ax1 = sns.scatterplot(x = data['smoothness_mean'], y = data['texture_mean'], hue = "diagnosis",
                    data = data, palette =palette, edgecolor=edgecolor)
plt.title('smoothness mean vs texture mean')
plt.subplot(222)
ax2 = sns.scatterplot(x = data['radius_mean'], y = data['fractal_dimension_worst'], hue = "diagnosis",
                    data = data, palette =palette, edgecolor=edgecolor)
plt.title('radius mean vs fractal dimension_worst')
plt.subplot(223)
ax3 = sns.scatterplot(x = data['texture_mean'], y = data['symmetry_mean'], hue = "diagnosis",
                    data = data, palette =palette, edgecolor=edgecolor)
plt.title('texture mean vs symmetry mean')
plt.subplot(224)
ax4 = sns.scatterplot(x = data['texture_mean'], y = data['symmetry_se'], hue = "diagnosis",
                    data = data, palette =palette, edgecolor=edgecolor)
plt.title('texture mean vs symmetry se')

fig.suptitle('Uncorrelated features', fontsize = 20)
plt.savefig('2')
plt.show()
```
![Uncorrelated features](/images/breast_cancer/breast_cancer20.jpg)

<a id="ch9"></a>
# Step 7: Negative correlated features

```
fig = plt.figure(figsize=(12,12))

plt.subplot(221)
ax1 = sns.scatterplot(x = data['area_mean'], y = data['fractal_dimension_mean'], hue = "diagnosis",
                    data = data, palette =palette, edgecolor=edgecolor)
plt.title('smoothness mean vs fractal dimension mean')
plt.subplot(222)
ax2 = sns.scatterplot(x = data['radius_mean'], y = data['fractal_dimension_mean'], hue = "diagnosis",
                    data = data, palette =palette, edgecolor=edgecolor)
plt.title('radius mean vs fractal dimension mean')
plt.subplot(223)
ax2 = sns.scatterplot(x = data['area_mean'], y = data['smoothness_se'], hue = "diagnosis",
                    data = data, palette =palette, edgecolor=edgecolor)
plt.title('area mean vs fractal smoothness se')
plt.subplot(224)
ax2 = sns.scatterplot(x = data['smoothness_se'], y = data['perimeter_mean'], hue = "diagnosis",
                    data = data, palette =palette, edgecolor=edgecolor)
plt.title('smoothness se vs perimeter mean')

fig.suptitle('Negative correlated features', fontsize = 20)
plt.savefig('3')
plt.show()
```
![Negative correlated features](/images/breast_cancer/breast_cancer21.jpg)

<a id="ch90"></a>
# References
I would like to express gratitude for the following resources, and thank developers for the inspiration:

* [Breast Cancer Analysis and Prediction](https://www.kaggle.com/code/vincentlugat/breast-cancer-analysis-and-prediction) - Indepth dive to EDA steps.
* [Advanced EDA](https://www.kaggle.com/code/kanberburak/advanced-eda) -  Advanced EDA




