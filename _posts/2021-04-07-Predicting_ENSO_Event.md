---
layout: post
title:  "Predicting ENSO Event"
date:   2021-04-07 7:55:10 -0700
---

```
# from google.colab import files
# To check existence of files
import os

# The following allows us to read files from Google Drive
from google.colab import drive
drive.mount('/content/gdrive')

%matplotlib inline

import xarray as xr
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import sklearn
import sklearn.ensemble
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import scipy.stats
from scipy.stats import pearsonr
from tqdm import tqdm
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

!pip install netCDF4

# Display YouTube 
from IPython.display import YouTubeVideo
```

    Mounted at /content/gdrive
    Collecting netCDF4
    [?25l  Downloading https://files.pythonhosted.org/packages/37/56/f65978898fb8e7e5df9c67531d86eb24eb04938deae3b61dbcce12c98212/netCDF4-1.5.6-cp37-cp37m-manylinux2014_x86_64.whl (4.7MB)
    [K     |████████████████████████████████| 4.7MB 5.7MB/s 
    [?25hCollecting cftime
    [?25l  Downloading https://files.pythonhosted.org/packages/41/e0/3e120cca16571c5ee3b35f1ed432c2aae5dc91e2b789e8b9c3a70e721ea0/cftime-1.4.1-cp37-cp37m-manylinux2014_x86_64.whl (313kB)
    [K     |████████████████████████████████| 317kB 49.2MB/s 
    [?25hRequirement already satisfied: numpy>=1.9 in /usr/local/lib/python3.7/dist-packages (from netCDF4) (1.19.5)
    Installing collected packages: cftime, netCDF4
    Successfully installed cftime-1.4.1 netCDF4-1.5.6


# Predicting the ENSO Event
This notebook goes over a short elementary explanation of the El Niño-Southern Oscillation (ENSO) phenomenon as well as a gentle introduction to machine learning tools used to predict an ENSO event.

## Theoretical Background Part I

Generally speaking, the ENSO phenomenon is a special climate pattern related to sea surface temperature and air pressure. The names El Niño and Southern Oscillation imply there are two components, the former is oceanic and the latter atmospheric. The term El Niño is used to refer to warmer than usual sea surface temperatures, while Southern Oscillation indicates shifts of mass in the tropical atmosphere. The opposite to an El Niño event (colder than usual temperatures) is denoted by La Niña. 

Historically, the label of El Niño was coined by fishermen in Peru alluding to Child Jesus since this phenomenon was observed around Christmas Holiday. The effects of El Niño are relevant to peruvians and their local economy since warmer water in the Pacific coast implies a decrease in fish population. Now we understand the gravity of such an event on a global scale due to a direct link to rainfall, temperature, vegetation, environmental anomalies, and disease outbreaks. Figures 1 and 2 are taken from [Pierre Madl](http://biophysics.sbg.ac.at/atmo/elnino.htm), a recommended read for those interested in more details regarding the ENSO phenomenon.

<figure>
<center>
<img src='https://drive.google.com/uc?export=view&id=1GHURs8ZJzBdeCEPuz451cILx4d0WEnPO' />
<figcaption> Figure 1: Normal wind conditions. </figcaption></center>
</figure>

<figure>
<center>
<img src='https://drive.google.com/uc?export=view&id=1TEzsFQ665nX5CIOQvA0GTI-1rXp4j_Ef' />
<figcaption> Figure 2: ENSO wind conditions. </figcaption></center>
</figure>

Next we mention some useful metrics to define El Niño or La Niña events, and a [Niño Index Region map](https://climatedataguide.ucar.edu/climate-data/nino-sst-indices-nino-12-3-34-4-oni-and-tni) taken from the Climate Data Guide by the University Corporation for Atmospheric Research (UCAR).
* SST (Sea Surface Temperature): This is measured directly from telemetric data.
* SOI (Southern Oscillation Index): Difference in pressure, between two measuring stations, at maximum or minimum value.
* Niño Index Regions: Niño1+2, Niño3, Niño3.4, Niño4, ONI.
* Indices are based on SST anomalies over a given index region.

<figure>
<center>
<img src='https://drive.google.com/uc?export=view&id=1p5giXN4CzfrXXE80h7msnVL2qxi_h8FN' />
<figcaption> Figure 3: Niño Index Regions. </figcaption></center>
</figure>

Those interested in visualizing sea surface temperature anomalies throughout the years are referred to the [SST Anomaly Timeline](https://svs.gsfc.nasa.gov/4695) provided by the NASA Scientific Visualization Studio.




```
YouTubeVideo('fVE7aH8Patw')
```





        <iframe
            width="400"
            height="300"
            src="https://www.youtube.com/embed/fVE7aH8Patw"
            frameborder="0"
            allowfullscreen
        ></iframe>
        



## Theoretical Background Part II 

Now, on to machine learning. We begin by motivating the tools required to allow for a computer to learn the best relationship between an input $x$ and output $f$. Once a relationship is deemed optimal, we make use of the optimized relationship to make predictions based on input data unknown to our model.

We will consider the following scenario. Assume we have a collection of $K$ images of cats, dogs, and ships. We  denote the set of images $X$. In mathematical terms, the cardinality of $X$ is
$$
| X | = K.
$$
A given image in the set $X$ will be identified by $x_i$ for $i \in \{0,1,2,3,\ldots,K-1\}$. Now suppose a computer is given the task of classifying each image as either a cat, dog, or ship. This implies the output space is the discrete set $Y = \{0,1,2\}$ where we take the bijection
$$
\begin{pmatrix} 
\text{Cat} \\ \text{Dog} \\ \text{Ship}
\end{pmatrix} 
\leftrightarrow
\begin{pmatrix}
0 \\ 1 \\ 2
\end{pmatrix}.
$$

An element of $Y$ will be denoted by $y_{i} \in \{0,1,2\}$ for some $i$ dependent on the image in question. To relate an input image with a class (cat, dog, or ship), we consider the $2$-tuple $(x_i,y_i)$. We also need knowledge of a score vector $f \in \mathbb{R}^{3}$. The score vector will be obtained via a classifier matrix, $W$, and a bias vector, $b$, described by 
$$
f(x,W) = Wx+b.
$$

Let's look at a toy example taken from the [Stanford CS231n course](http://cs231n.stanford.edu/) to explore the roles of $W$ and $b$. Assume the images are of size $2 {\times} 2$ for simplicity and denote the space of $m$-by-$n$ matrices by $M_{m {\times} n}$. 
<figure>
<center>
<img src='https://drive.google.com/uc?export=view&id=16bp2hvLGgg7JzNid-B1qGnJzyRvB1viJ' />
<figcaption> Figure 4: Visual Representation of a Discrete Classifier. Source: Stanford CS231n Course. </figcaption></center>
</figure>

* The input image $x$ belongs to the space of two-by-two matrices, which is isomorphic to the space of four-by-one column vectors, i.e., $\forall i \in \{0,\ldots,K-1\},x_i \in X \subset M_{2 {\times} 2}  \cong M_{4 {\times} 1} $
* The discrete classifier is $W \in M_{3 {\times} 4}$ where each row represents a specific class. The amount of columns coincide with the size of the input image, else the matrix multiplcation is not defined.
* The bias vector, $b \in M_{3 {\times} 1}$, is that which is added to $Wx$ depending on the distribution of images in our dataset, e.g., if there is knowledge of a majority of cat pictures in $X$ then we may adjust $b$ accordingly. 
* The output is our score vector $f$, represented as a column vector in the space $M_{3 {\times} 1}$. This vector represents how certain the machine is that a given input image belongs to a specific class. In our example, the machine is "confident" the image is that of a dog, less "confident" the image is that of a ship, and very "confident" the image does not represent a cat.

**Remark**: The bias vector $b$ can be encoded in the matrix $W$ if we are clever about the dimensions in which we work in. We will focus on the model $f(x,W) = Wx$.

So, we have described a model that classifies images incorrectly. What can we fix? We can do a better job at choosing $W$. This is the primary goal of machine learning. We need to quantify how good our choice of $W$ by introducing a *loss function*. Before we proceed we recall that our goal is to predict a value from a discrete set, $Y$. This is not always the case. There are scenarios where we wish to predict values from a continuous set. The discrete classification problem involves *classification loss*, whereas the continuous case deals with *regression loss*.

Moving forward with classification loss, we define *Multiclass Support Vector Machine loss* for a given image as
$$
L_{i} = \sum_{j \neq y_{i}}^{|Y|-1} \max(0,s_j-s_{y_i}+\Delta)
$$
where $s_j$ denotes the $j$-th component of $f$, and $\Delta \in \mathbb{R}_{+}$ is some fixed margin (we won't worry about choosing $\Delta$ here).

Next, let's look at an example for a given image and output vector. Assume we pick an image of a dog from our dataset. This means that for some fixed positive integer $k$, we have $(x_k,y_k) = (x_k,1)$. Also, assume that for a given classifier $W$, we have the following 
$$
s = f(x_{k},W) = Wx_{k} = \begin{pmatrix} 8 \\ 10 \\ -10 \end{pmatrix}.
$$

Using $\Delta = 3$ (again, this is just a threshold value so no need to pay much attention) and $s_{y_1} = s_{1} = 10$, the loss (or cost) for the selected image is:
\begin{align*}
L_k & =  \sum_{j \neq y_k}^{|Y|-1} \max(0,s_j-s_{y_k}+\Delta) \\
& = \sum_{j \neq 1}^{2} \max(0,s_{j}-s_{1} + \Delta) \\
& = \sum_{j \neq 1}^{2} \max(0,s_{j}-10 + \Delta) \\
& = \max(0,s_{0}-10+\Delta) + \max(0,s_{2}-10+\Delta) \\
& = \max(0,8-10+\Delta) + \max(0,-10-10+\Delta) \\
& = \max(0,1) + \max(0,-17) \\
& = 1 + 0 \\
\therefore L_{k} & = 1 .
\end{align*}

As a summary:
* We have calculated the SVM loss, $L_i$, for a given image.
* If zero is selected from the $\max$ function this means we are confident a given score, $s_j$, and the correct score, $s_{y_{0}}$, are far from each other so we do not accumulate loss.
*  If a nonzero value is selected from the $\max$ function this means we are not confident in distinguishing between a given score, $s_j$, and the score of the correct class, $s_{y_0}$, so we accumulate loss.

Calculating the loss over the entire dataset and taking the average yields
$$
L = \frac{1}{K} \sum_{i=0}^{K-1} L_{i}.
$$
We pay closer attention to $L$ as a function of $W$, since the score depends on the classifier. In other words,
\begin{align*}
L(W) & = \frac{1}{K} \sum_{i=0}^{K-1} ( L_{i} ) \\
& = \frac{1}{K} \sum_{i=0}^{K-1} \left( \sum_{j \neq y_{i}}^{|Y|-1} \max(0,s_j-s_{y_i}+\Delta) \right) \\
& = \frac{1}{K} \sum_{i=0}^{K-1} \left( \sum_{j \neq y_{i}}^{|Y|-1} \max(0,f(x_i,W)_j-f(x_i,W)_{y_i}+\Delta) \right)
\end{align*}

One last detail is that of regularization. Suppose we had the following image $x$, and row vectors $w_1$, $w_2$ of the classifier matrix $W$:
\begin{align*}
x = \begin{pmatrix}1 \\ 1 \\ 1 \\ 1 \end{pmatrix}, \quad
w_1 = \begin{pmatrix} 1 \\ 0 \\ 0 \\ 0 \end{pmatrix}^{T}, \quad
w_{2} \begin{pmatrix} \frac{1}{4} \\ \frac{1}{4} \\ \frac{1}{4} \\ \frac{1}{4} \end{pmatrix}^{T}.
\end{align*}
We see that the dot product between our image and the rows of $W$ is not unique, i.e., $\langle w_1^{T}, x \rangle = \langle w_2^{T}, x \rangle$. So how do we favor a row of $W$ to classify our image? This is done by including the regularization term 
$$
R(W) = \sum_{p} \sum_{q} (W_{p,q})^{2}
$$
in our loss function. Here, $p$ and $q$ sum over the rows and columns of $W$ respectively. Finally, by introducing a parameter $\lambda$ to tune our regularization, we obtain
$$
L(W) = \frac{1}{K} \sum_{i=0}^{K-1} ( L_{i} ) + \lambda R(W).
$$
Generally speaking, $\lambda$ is inversely proportional to the amount of miss-classifications allowed. For large values of $\lambda$, less wrong classified examples are allowed. For smaller $\lambda$, more miss-classifications are allowed.

**Remark**: There exists loss functions which give more intuitive values. For example, cross-entropy loss yields a vector with values in the interval $I = (0,1)$ . This loss function has a probabilistic point of view.

Now, the million-dollar question is: How to optimize our loss function, $L(W)$?. This can be achieved via gradient descent algorithms. At San Diego State University, these algorithms are explored in the course MATH 693-A. 

Before getting into the numerics regarding the prediction of an ENSO event, we conclude this section by listing two main ideas:
* Given a dataset with the known relationship $(x_i,y_i)$ how can we train a computer to predict $y_i$ for an $x$ which is not in our set $X$ (unseen data).
* We require the notion of a loss function which will be optimized so we can choose an appropriate classifier $W$.


## Numerics (Hackathon)

Now on to numerics. We will revisit code provided in the notebook *AI for Earth System Science Hackathon 2020: Seasonal Forecasting* authored by Ankur Mahesh from ClimateAi. Those interested in the original notebook, or other similar hackathon challenges are referred to the following [repository](https://github.com/NCAR/ai4ess-hackathon-2020).

Akin to our previous example where the input and output data corresponded to an image and a discrete class respectively, here we have:
* Input data corresponds to historical sea surface temperatures found in the [COBE-SST dataset](https://psl.noaa.gov/data/gridded/data.cobe.html).
* Output data corresponds to [Nino3.4 indices](https://www.ncdc.noaa.gov/teleconnections/enso/indicators/sst/) at a given lead time. 

**Remark**: We skipped a discussion on training, validation, and testing sets. Briefly explained, our model will initially only interact with what we denote training and validation datasets. The training set, as the name explains, trains our model while the validation provides insight into the performance of our model. It is only until the last step of our machine learning process that our trained model will interact with the test set.  

Without further ado, let's obtain the required dataset with a few lines of code.



```
# COBE-SST dataset
!wget http://portal.nersc.gov/project/dasrepo/AGU_ML_Tutorial/sst.mon.mean.trefadj.anom.1880to2018.nc
# Niño3.4 Indices
!wget http://portal.nersc.gov/project/dasrepo/AGU_ML_Tutorial/nino34.long.anom.data.txt
```

    --2021-04-02 18:40:45--  http://portal.nersc.gov/project/dasrepo/AGU_ML_Tutorial/sst.mon.mean.trefadj.anom.1880to2018.nc
    Resolving portal.nersc.gov (portal.nersc.gov)... 128.55.206.24, 128.55.206.28
    Connecting to portal.nersc.gov (portal.nersc.gov)|128.55.206.24|:80... connected.
    HTTP request sent, awaiting response... 301 Moved Permanently
    Location: https://portal.nersc.gov/project/dasrepo/AGU_ML_Tutorial/sst.mon.mean.trefadj.anom.1880to2018.nc [following]
    --2021-04-02 18:40:46--  https://portal.nersc.gov/project/dasrepo/AGU_ML_Tutorial/sst.mon.mean.trefadj.anom.1880to2018.nc
    Connecting to portal.nersc.gov (portal.nersc.gov)|128.55.206.24|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 432481041 (412M) [application/x-netcdf]
    Saving to: ‘sst.mon.mean.trefadj.anom.1880to2018.nc’
    
    sst.mon.mean.trefad 100%[===================>] 412.45M  20.1MB/s    in 27s     
    
    2021-04-02 18:41:13 (15.5 MB/s) - ‘sst.mon.mean.trefadj.anom.1880to2018.nc’ saved [432481041/432481041]
    
    URL transformed to HTTPS due to an HSTS policy
    --2021-04-02 18:41:13--  https://portal.nersc.gov/project/dasrepo/AGU_ML_Tutorial/nino34.long.anom.data.txt
    Resolving portal.nersc.gov (portal.nersc.gov)... 128.55.206.24, 128.55.206.28
    Connecting to portal.nersc.gov (portal.nersc.gov)|128.55.206.24|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 15449 (15K) [text/plain]
    Saving to: ‘nino34.long.anom.data.txt’
    
    nino34.long.anom.da 100%[===================>]  15.09K  --.-KB/s    in 0.06s   
    
    2021-04-02 18:41:13 (266 KB/s) - ‘nino34.long.anom.data.txt’ saved [15449/15449]
    


Next, we make use of Ankur's code to reshape our dataset with appropriate dimensions and define functions which will access data in adequate fashion. It is also worthwhile to mention the use of Principal Component Analysis to reduce the dimension of our problem if necessary. This is a great tool when working with large datasets.  


```
#Scaffold code to load in data.  This code cell is mostly data wrangling

def load_enso_indices():
  """
  Reads in the txt data file to output a pandas Series of ENSO vals

  outputs
  -------

    pd.Series : monthly ENSO values starting from 1870-01-01
  """
  with open('nino34.long.anom.data.txt') as f:
    line = f.readline()
    enso_vals = []
    while line:
        yearly_enso_vals = map(float, line.split()[1:]) 
        enso_vals.extend(yearly_enso_vals)
        line = f.readline()

  enso_vals = pd.Series(enso_vals)
  enso_vals.index = pd.date_range('1870-01-01',freq='MS',
                                  periods=len(enso_vals))
  enso_vals.index = pd.to_datetime(enso_vals.index)
  return enso_vals

def assemble_basic_predictors_predictands(start_date, end_date, lead_time, 
                                    use_pca=False, n_components=32):
  """
  inputs
  ------

      start_date        str : the start date from which to extract sst
      end_date          str : the end date 
      lead_time         str : the number of months between each sst
                              value and the target Nino3.4 Index
      use_pca          bool : whether or not to apply principal components
                              analysis to the sst field
      n_components      int : the number of components to use for PCA

  outputs
  -------
      Returns a tuple of the predictors (np array of sst temperature anomalies) 
      and the predictands (np array the ENSO index at the specified lead time).

  """
  ds = xr.open_dataset('sst.mon.mean.trefadj.anom.1880to2018.nc')
  sst = ds['sst'].sel(time=slice(start_date, end_date)) 
  num_time_steps = sst.shape[0]
  
  #sst is a 3D array: (time_steps, lat, lon)
  #in this tutorial, we will not be using ML models that take
  #advantage of the spatial nature of global temperature
  #therefore, we reshape sst into a 2D array: (time_steps, lat*lon)
  #(At each time step, there are lat*lon predictors)
  sst = sst.values.reshape(num_time_steps, -1)
  sst[np.isnan(sst)] = 0

  #Use Principal Components Analysis, also called
  #Empirical Orthogonal Functions, to reduce the
  #dimensionality of the array
  if use_pca:
    pca = sklearn.decomposition.PCA(n_components=n_components)
    pca.fit(sst)
    X = pca.transform(sst)
  else:
    X = sst

  start_date_plus_lead = pd.to_datetime(start_date) + \
                        pd.DateOffset(months=lead_time)
  end_date_plus_lead = pd.to_datetime(end_date) + \
                      pd.DateOffset(months=lead_time)
  y = load_enso_indices()[slice(start_date_plus_lead, 
                                end_date_plus_lead)]


  ds.close()
  return X, y

def plot_nino_time_series(y, predictions, title):
  """
  inputs
  ------
    y           pd.Series : time series of the true Nino index
    predictions np.array  : time series of the predicted Nino index (same
                            length and time as y)
    titile                : the title of the plot

  outputs
  -------
    None.  Displays the plot
  """
  predictions = pd.Series(predictions, index=y.index)
  predictions = predictions.sort_index()
  y = y.sort_index()

  plt.plot(y, label='Ground Truth')
  plt.plot(predictions, '--', label='ML Predictions')
  plt.legend(loc='best')
  plt.title(title)
  plt.ylabel('Nino3.4 Index')
  plt.xlabel('Date')
  plt.show()
  plt.close()
```

Now we store our training and validation sets with appropriate names.


```
# Sample loading of data

# Training Set
X_train, y_train = assemble_basic_predictors_predictands('1980-01-01','1995-12-31', lead_time=1)

# Validation Set
X_val, y_val = assemble_basic_predictors_predictands('1997-01-01','2006-12-31', lead_time=1)

# Sanity check:
print(X_train.shape)
print(X_train[0,0:5])
print(y_train.shape)
print(y_train[0:5])
```

    (192, 64800)
    [-0.00451612 -0.00451612 -0.00451612 -0.00451612 -0.00451612]
    (192,)
    1980-02-01    0.33
    1980-03-01    0.09
    1980-04-01    0.20
    1980-05-01    0.13
    1980-06-01    0.37
    Freq: MS, dtype: float64


At this point we steer away from classification loss and turn our attention to regression loss since we want to predict Niño3.4 indices (values from a cotinuous set). The mathematics behind the scenes still involves optimizing some loss function, it's just distinct to SVM loss previously defined. 


```
# Here, the ground truth corresponds to y_val (validation set)
# Our program produces predictions based on only
# the training data (X_train, y_train), and the predictor
# (X_val)

#Let's use a linear regression model
regr = sklearn.linear_model.LinearRegression()

# (X_train, y_train) is information we have as input
# This data trains the model 
regr.fit(X_train,y_train)

# We predict outputs based on our trained model and X_val
predictions = regr.predict(X_val)

# Compare our prediction versus y_val and
# compute metric for quality of predictions
# RMSE: Root Mean Square Error 
# corr: Pearson Correlation Coefficient
corr, _ = scipy.stats.pearsonr(predictions, y_val)
rmse = mean_squared_error(y_val, predictions)
print("RMSE: {:.2f}".format(rmse))

plot_nino_time_series(y_val, predictions, 
    'Predicted and True Nino3.4 Indices on Training Set at 1 Month Lead Time. \n Corr: {:.2f}'.format(corr))
```

    RMSE: 0.28



![png](Predicting_ENSO_Event_files/Predicting_ENSO_Event_12_1.png)


## Numerics (NOAA Project)

We wish to update the previous code to accommodate the [GODAS dataset](https://www.cpc.ncep.noaa.gov/products/GODAS/) and our computed temperature anomalies. We have already taken the liberty of combining the yearly GODAS files into a single one. 

Moreover, we have modified the code shown in the previous section allowing us to read files located in our Google Drive so please be careful with the correct directory. 


```
def load_enso_indices():
  """
  Reads in the txt data file to output a pandas Series of ENSO vals

  outputs
  -------

    pd.Series : monthly ENSO values starting from 1870-01-01
  """

  with open(anomaly_path) as f:
    line = f.readline()
    enso_vals = []
    while line:
        yearly_enso_vals = map(float, line.split()[1:]) 
        enso_vals.extend(yearly_enso_vals)
        line = f.readline()

  enso_vals = pd.Series(enso_vals)
  enso_vals.index = pd.date_range('1980-01-01',freq='MS',
                                  periods=len(enso_vals))
  enso_vals.index = pd.to_datetime(enso_vals.index)
  return enso_vals

def assemble_basic_predictors_predictands(start_date, end_date, lead_time,
                                    use_pca=False, n_components=32):
  """
  inputs
  ------

      start_date        str : the start date from which to extract sst
      end_date          str : the end date 
      lead_time         str : the number of months between each sst
                              value and the target Nino3.4 Index
      use_pca          bool : whether or not to apply principal components
                              analysis to the sst field
      n_components      int : the number of components to use for PCA

  outputs
  -------
      Returns a tuple of the predictors (np array of sst temperature anomalies) 
      and the predictands (np array the ENSO index at the specified lead time).

  """
  ds = xr.open_dataset(godas_path)
  sst = ds['deepTemp'].sel(time=slice(start_date, end_date)) 
  num_time_steps = sst.shape[0]

  sst = sst.values.reshape(num_time_steps, -1)
  sst[np.isnan(sst)] = 0

  if use_pca:
    pca = sklearn.decomposition.PCA(n_components=n_components)
    pca.fit(sst)
    X = pca.transform(sst)
  else:
    X = sst

  start_date_plus_lead = pd.to_datetime(start_date) + \
                        pd.DateOffset(months=lead_time)
  end_date_plus_lead = pd.to_datetime(end_date) + \
                      pd.DateOffset(months=lead_time)
  y = load_enso_indices()[slice(start_date_plus_lead, 
                                end_date_plus_lead)]


  ds.close()
  return X, y

def plot_nino_time_series(y, predictions, depth, title):
  """
  inputs
  ------
    y           pd.Series : time series of the true Nino index
    predictions np.array  : time series of the predicted Nino index (same
                            length and time as y)
    title                 : the title of the plot
    depth                 : depth level being considered

  outputs
  -------
    None.  Displays the plot
  """
  images_dir = '/content/gdrive/My Drive/ColabNotebooks/Figures/GODAS_Regression'
  figName_png = str(depth)+'m_Predictions.png'
  figName_pdf = str(depth)+'m_Predictions.pdf'

  predictions = pd.Series(predictions, index=y.index)
  predictions = predictions.sort_index()
  y = y.sort_index()

  plt.plot(y, label='Ground Truth')
  plt.plot(predictions, '--', label='ML Predictions')
  plt.legend(loc='best')
  plt.title(title)
  plt.ylabel(str(depth)+ " " + 'm Temperature Anomalies')
  plt.xlabel('Date')
  plt.savefig(f"{images_dir}"+"/"+figName_png, bbox_inches='tight')
  plt.savefig(f"{images_dir}"+"/"+figName_pdf, bbox_inches='tight')
#  plt.show()
  plt.close()
```


```
# Depth levels we consider
depth_arr = np.arange(105,215,10)

for ii in range(len(depth_arr)):
  depth_sel = depth_arr[ii]
  # Read data from Drive
  godas_path = "/content/gdrive/MyDrive/ColabNotebooks/Yearly_NC_Files_Plus_Anomalies/godasData_"+str(depth_sel)+"m.nc"
  anomaly_path = "/content/gdrive/MyDrive/ColabNotebooks/Yearly_NC_Files_Plus_Anomalies/deepTemp_anomalies_"+str(depth_sel)+"m.txt"
  # Select lead time 
  month_lead = 6
  # Training and validation data sets
  X_train, y_train = assemble_basic_predictors_predictands('1980-01-01','1995-12-31', lead_time=month_lead)                                                         
  X_val, y_val = assemble_basic_predictors_predictands('1997-01-01','2006-12-31', lead_time=month_lead)

  regr = sklearn.linear_model.LinearRegression()
  regr.fit(X_train,y_train)
  predictions = regr.predict(X_val)
  corr, _ = scipy.stats.pearsonr(predictions, y_val)
  # rmse = mean_squared_error(y_val, predictions)
  # print("RMSE: {:.2f}".format(rmse))

  # Save figure
  plot_nino_time_series(y_val, predictions, depth_sel,
      'Predicted and True Ocean Temperature Anomalies at'+' '+str(month_lead)+' '+'Month Lead Time. \n Corr: {:.2f}'.format(corr))
```


```
%%shell
jupyter nbconvert --to Markdown /content/Predicting_ENSO_Event.ipynb
```

    [NbConvertApp] WARNING | pattern u'/content/Predicting_ENSO_Event.ipynb' matched no files
    This application is used to convert notebook files (*.ipynb) to various other
    formats.
    
    WARNING: THE COMMANDLINE INTERFACE MAY CHANGE IN FUTURE RELEASES.
    
    Options
    -------
    
    Arguments that take values are actually convenience aliases to full
    Configurables, whose aliases are listed on the help line. For more information
    on full configurables, see '--help-all'.
    
    --execute
        Execute the notebook prior to export.
    --allow-errors
        Continue notebook execution even if one of the cells throws an error and include the error message in the cell output (the default behaviour is to abort conversion). This flag is only relevant if '--execute' was specified, too.
    --no-input
        Exclude input cells and output prompts from converted document. 
        This mode is ideal for generating code-free reports.
    --stdout
        Write notebook output to stdout instead of files.
    --stdin
        read a single notebook file from stdin. Write the resulting notebook with default basename 'notebook.*'
    --inplace
        Run nbconvert in place, overwriting the existing notebook (only 
        relevant when converting to notebook format)
    -y
        Answer yes to any questions instead of prompting.
    --clear-output
        Clear output of current file and save in place, 
        overwriting the existing notebook.
    --debug
        set log level to logging.DEBUG (maximize logging output)
    --no-prompt
        Exclude input and output prompts from converted document.
    --generate-config
        generate default config file
    --nbformat=<Enum> (NotebookExporter.nbformat_version)
        Default: 4
        Choices: [1, 2, 3, 4]
        The nbformat version to write. Use this to downgrade notebooks.
    --output-dir=<Unicode> (FilesWriter.build_directory)
        Default: ''
        Directory to write output(s) to. Defaults to output to the directory of each
        notebook. To recover previous default behaviour (outputting to the current
        working directory) use . as the flag value.
    --writer=<DottedObjectName> (NbConvertApp.writer_class)
        Default: 'FilesWriter'
        Writer class used to write the  results of the conversion
    --log-level=<Enum> (Application.log_level)
        Default: 30
        Choices: (0, 10, 20, 30, 40, 50, 'DEBUG', 'INFO', 'WARN', 'ERROR', 'CRITICAL')
        Set the log level by value or name.
    --reveal-prefix=<Unicode> (SlidesExporter.reveal_url_prefix)
        Default: u''
        The URL prefix for reveal.js (version 3.x). This defaults to the reveal CDN,
        but can be any url pointing to a copy  of reveal.js.
        For speaker notes to work, this must be a relative path to a local  copy of
        reveal.js: e.g., "reveal.js".
        If a relative path is given, it must be a subdirectory of the current
        directory (from which the server is run).
        See the usage documentation
        (https://nbconvert.readthedocs.io/en/latest/usage.html#reveal-js-html-
        slideshow) for more details.
    --to=<Unicode> (NbConvertApp.export_format)
        Default: 'html'
        The export format to be used, either one of the built-in formats
        ['asciidoc', 'custom', 'html', 'latex', 'markdown', 'notebook', 'pdf',


        'python', 'rst', 'script', 'slides'] or a dotted object name that represents
        the import path for an `Exporter` class
    --template=<Unicode> (TemplateExporter.template_file)
        Default: u''
        Name of the template file to use
    --output=<Unicode> (NbConvertApp.output_base)
        Default: ''
        overwrite base name use for output files. can only be used when converting
        one notebook at a time.
    --post=<DottedOrNone> (NbConvertApp.postprocessor_class)
        Default: u''
        PostProcessor class used to write the results of the conversion
    --config=<Unicode> (JupyterApp.config_file)
        Default: u''
        Full path of a config file.
    
    To see all available configurables, use `--help-all`
    
    Examples
    --------
    
        The simplest way to use nbconvert is
        
        > jupyter nbconvert mynotebook.ipynb
        
        which will convert mynotebook.ipynb to the default format (probably HTML).
        
        You can specify the export format with `--to`.
        Options include ['asciidoc', 'custom', 'html', 'latex', 'markdown', 'notebook', 'pdf', 'python', 'rst', 'script', 'slides'].
        
        > jupyter nbconvert --to latex mynotebook.ipynb
        
        Both HTML and LaTeX support multiple output templates. LaTeX includes
        'base', 'article' and 'report'.  HTML includes 'basic' and 'full'. You
        can specify the flavor of the format used.
        
        > jupyter nbconvert --to html --template basic mynotebook.ipynb
        
        You can also pipe the output to stdout, rather than a file
        
        > jupyter nbconvert mynotebook.ipynb --stdout
        
        PDF is generated via latex
        
        > jupyter nbconvert mynotebook.ipynb --to pdf
        
        You can get (and serve) a Reveal.js-powered slideshow
        
        > jupyter nbconvert myslides.ipynb --to slides --post serve
        
        Multiple notebooks can be given at the command line in a couple of 
        different ways:
        
        > jupyter nbconvert notebook*.ipynb
        > jupyter nbconvert notebook1.ipynb notebook2.ipynb
        
        or you can specify the notebooks list in a config file, containing::
        
            c.NbConvertApp.notebooks = ["my_notebook.ipynb"]
        
        > jupyter nbconvert --config mycfg.py
    



    ---------------------------------------------------------------------------

    CalledProcessError                        Traceback (most recent call last)

    <ipython-input-91-b08d09e59ca5> in <module>()
    ----> 1 get_ipython().run_cell_magic('shell', '', 'jupyter nbconvert --to Markdown /content/Predicting_ENSO_Event.ipynb')
    

    /usr/local/lib/python3.7/dist-packages/IPython/core/interactiveshell.py in run_cell_magic(self, magic_name, line, cell)
       2115             magic_arg_s = self.var_expand(line, stack_depth)
       2116             with self.builtin_trap:
    -> 2117                 result = fn(magic_arg_s, cell)
       2118             return result
       2119 


    /usr/local/lib/python3.7/dist-packages/google/colab/_system_commands.py in _shell_cell_magic(args, cmd)
        111   result = _run_command(cmd, clear_streamed_output=False)
        112   if not parsed_args.ignore_errors:
    --> 113     result.check_returncode()
        114   return result
        115 


    /usr/local/lib/python3.7/dist-packages/google/colab/_system_commands.py in check_returncode(self)
        137     if self.returncode:
        138       raise subprocess.CalledProcessError(
    --> 139           returncode=self.returncode, cmd=self.args, output=self.output)
        140 
        141   def _repr_pretty_(self, p, cycle):  # pylint:disable=unused-argument


    CalledProcessError: Command 'jupyter nbconvert --to Markdown /content/Predicting_ENSO_Event.ipynb' returned non-zero exit status 255.



```

```
