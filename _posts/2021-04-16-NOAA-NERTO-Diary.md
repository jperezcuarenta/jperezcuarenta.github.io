---
layout: post
title:  "NOAA NERTO Diary"
author: "Jesus Perez Cuarenta"
categories: journal
tags: [documentation]
---

One of my current projects involves weekly reporting to folks at NOAA Cooperative Science Center in Atmospheric Sciences and Meteorology (NCAS-M) as well as [Dr. Samuel Shen](https://shen.sdsu.edu/) (faculty advisor) and [Dr. Thomas Smith](https://www.star.nesdis.noaa.gov/star/Smith_TM.php) (NOAA mentor). This post will serve as a diary for documenting my progress.

First things first: Python libraries and access to Google Drive.


```python
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
!pip install netCDF4
```

## Week 1

My research revolves around predicting El Niño-Southern Oscillation events by means of machine learning algorithms influenced by code and results from the 2020 [AI for Earth System Science Hackathon](https://github.com/NCAR/ai4ess-hackathon-2020). There is Python code available which reads the [COBE sea surface temperature data](https://psl.noaa.gov/data/gridded/data.cobe.html), together with temperature anomalies, and outputs predicted anomalies at a given lead time. 

The original Python code is authored by Ankur Mahesh from ClimateAi and has been adapted for my research along with my own Matlab programs in order to predict anomalies using the [GODAS data set](https://psl.noaa.gov/data/gridded/data.godas.html). The key difference between data sets is that the GODAS set accounts for deep ocean temperature data. One of our main goals is to shed light on the usefulness of predicting temperature anomalies with temperature at varying depth levels, rather than only at the surface. 

Initially I had to become familiar with [NetCDF](https://www.unidata.ucar.edu/software/netcdf/) so I could rewrite several yearly GODAS files as one. I have outlined the steps I took to achieve this in detail [here](https://jperezcuarenta.github.io/journal/Manipulating-NetCDF-in-Matlab.html).

I present Python code modified to work with the *.nc* file obtained in Matlab, as well as figures of the computed temperature anomalies. We see that the predictions are more meaningful at 105 meter depth than at 205 meters. In the second week of my research I will repeat the experiment with a more accurate data set provided by Dr. Shen and compare the results.


```python
# Modified code originally provided by Ankur Mahesh from ClimateAi.
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
  plt.close()
```


```python
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

Here are some of our results at 1 and and 6 month lead times.

* 1 Month Lead Time

<figure>
<center>
<img src='https://drive.google.com/uc?export=view&id=1-cw73E8VtvmE2rl-Wasrrw1AvOaSDDYP' />
<figcaption> Figure 5: Comparing 105 m deep ocean temperature anomalies at 1 month lead time. </figcaption></center>
</figure>

<figure>
<center>
<img src='https://drive.google.com/uc?export=view&id=10DE1WUG9mqifg3ZiBgaKLtUkPf9-7NZ_' />
<figcaption> Figure 6: Comparing 155 m deep ocean temperature anomalies at 1 month lead time. </figcaption></center>
</figure>

<figure>
<center>
<img src='https://drive.google.com/uc?export=view&id=10gzlDhJBlgQehBCgtjV2WSoq8XOKDXZE' />
<figcaption> Figure 7: Comparing 205 m deep ocean temperature anomalies at 1 month lead time. </figcaption></center>
</figure>

* 6 Month Lead Time

<figure>
<center>
<img src='https://drive.google.com/uc?export=view&id=12Lj1hnN1CAQN1TnV1JZqSv-A9TmZWG7P' />
<figcaption> Figure 8: Comparing 105 m deep ocean temperature anomalies at 6 month lead time. </figcaption></center>
</figure>

<figure>
<center>
<img src='https://drive.google.com/uc?export=view&id=13T1yooY6AP0a8SDxIGrzMKAVPGPYufP0' />
<figcaption> Figure 9: Comparing 155 m deep ocean temperature anomalies at 6 month lead time. </figcaption></center>
</figure>

<figure>
<center>
<img src='https://drive.google.com/uc?export=view&id=149kQmD4kUe7DHeV88hALudnmzrurPtqq' />
<figcaption> Figure 10: Comparing 205 m deep ocean temperature anomalies at 6 month lead time. </figcaption></center>
</figure>


