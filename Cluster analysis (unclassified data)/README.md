![Getting Started with a Movie Recommendation System](/images/movie/movie0.jpg)

This project focuses on movie recommendation systems, using TMDB 5000 Movie Dataset and gets inspiration from public notebooks.

# Table of Contents
1. [Chapter 1 - Project Overview](#ch1)
1. [Chapter 2 - Types of recommendation systems](#ch2)
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

The rapid growth of data collection has led to a new era of information. Data is being used to create more efficient systems and this is where Recommendation Systems come into play. Recommendation Systems are a type of information filtering systems as they improve the quality of search results and provides items that are more relevant to the search item or are realted to the search history of the user.

They are used to predict the rating or preference that a user would give to an item. Almost every major tech company has applied them in some form or the other: Amazon uses it to suggest products to customers, YouTube uses it to decide which video to play next on autoplay, and Facebook uses it to recommend pages to like and people to follow. Moreover, companies like Netflix and Spotify depend highly on the effectiveness of their recommendation engines for their business and sucees.

<a id="ch2"></a>
# Types of recommendation systems

There are basically three types of recommender systems:-

- Demographic Filtering- They offer generalized recommendations to every user, based on movie popularity and/or genre. The System recommends the same movies to users with similar demographic features. Since each user is different , this approach is considered to be too simple. The basic idea behind this system is that movies that are more popular and critically acclaimed will have a higher probability of being liked by the average audience.

- Content Based Filtering- They suggest similar items based on a particular item. This system uses item metadata, such as genre, director, description, actors, etc. for movies, to make these recommendations. The general idea behind these recommender systems is that if a person liked a particular item, he or she will also like an item that is similar to it.

- Collaborative Filtering- This system matches persons with similar interests and provides recommendations based on this matching. Collaborative filters do not require item metadata like its content-based counterparts.

<a id="ch3"></a>
# Step 1: Data Gathering and info from Kaggle

# About Dataset
Background
What can we say about the success of a movie before it is released? Are there certain companies (Pixar?) that have found a consistent formula? Given that major films costing over $100 million to produce can still flop, this question is more important than ever to the industry. Film aficionados might have different interests. Can we predict which films will be highly rated, whether or not they are a commercial success?

This is a great place to start digging in to those questions, with data on the plot, cast, crew, budget, and revenues of several thousand films.

# Data Source Transfer Summary
Kaggle have removed the original version of this dataset per a DMCA takedown request from IMDB. In order to minimize the impact, we're replacing it with a similar set of films and data fields from The Movie Database (TMDb) in accordance with their terms of use. The bad news is that kernels built on the old dataset will most likely no longer work.

New columns:

- homepage

- id

- original_title

- overview

- popularity

- production_companies

- production_countries

- release_date

- spoken_languages

- status

- tagline

- vote_average

Lost columns:

- actor1facebook_likes

- actor2facebook_likes

- actor3facebook_likes

- aspect_ratio

- casttotalfacebook_likes

- color

- content_rating

- directorfacebooklikes

- facenumberinposter

- moviefacebooklikes

- movieimdblink

- numcriticfor_reviews

- numuserfor_reviews

<a id="ch4"></a>

# Step 2: Head and describe

```
# Head
data.head()
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
# Two datasets
M = data[(data['diagnosis'] != 0)]
B = data[(data['diagnosis'] == 0)]
```

```
#------------COUNT-----------------------
trace = go.Bar(x = (len(M), len(B)), y = ['malignant', 'benign'], orientation = 'h', opacity = 0.8, marker=dict(
        color=[ 'gold', 'lightskyblue'],
        line=dict(color='#000000',width=1.5)))

layout = dict(title =  'Count of diagnosis variable')
                    
fig = dict(data = [trace], layout=layout)
py.iplot(fig)

#------------PERCENTAGE-------------------
trace = go.Pie(labels = ['benign','malignant'], values = data['diagnosis'].value_counts(), 
               textfont=dict(size=15), opacity = 0.8,
               marker=dict(colors=['lightskyblue', 'gold'], 
                           line=dict(color='#000000', width=1.5)))


layout = dict(title =  'Distribution of diagnosis variable')
           
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
    colors = ['#FFD700', '#7EC0EE']

    fig = ff.create_distplot(hist_data, group_labels, colors = colors, show_hist = True, bin_size = size_bin, curve_type='kde')
    
    fig['layout'].update(title = data_select)

    py.iplot(fig, filename = 'Density plot')
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

* [Getting Started with a Movie Recommendation System](https://www.kaggle.com/code/ibtesama/getting-started-with-a-movie-recommendation-system/notebook) 




