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

## Week 2

In Week 2 I began working with Dr. Shen's data set while referencing *A Dynamically Consistent Reconstruction of Ocean Temperature*, authored by Shen Et Al. 

I will mention some events, roadblocks, and results from this week. 

### Access to Data
In Week 1 I managed to find useful commands within Matlab to access yearly deep ocean temperature data automatically so I could avoid manually clicking at each download link. Now, the data set provided to me was stored in folders titled *5m, 100m, 20m, ..., 2000m*. After accessing the *100m* folder we see the following:
1. 100m1_Reconstructed_Temp_Anomaly_Jan1950.mat
2. 100m2_Reconstructed_Temp_Anomaly_Feb1950.mat
3. 100m3_Reconstructed_Temp_Anomaly_Mar1950.mat
4. ...

which culminates with *100m756_Reconstructed_Temp_Anomaly_Dec2012.mat* for a total of 756 *.mat* files. I took advantage of string concatenation, and conversion of numbers to strings, to loop through all file names and store as a single *.nc* file which may be useful for future students who wish to access the same data set with ease.

### Creating a Proper Time Vector
To accurately save data in an *.nc* file I also required vectors corresponding to longitude, latitude, and time. Creating longitude and latitude vectors was straight-forward. For the time component, I wanted a vector, $t$, which stored the days passed since 1950-01-01 (Year-Month-Day) in the $i$-th entry. I learned of the functions [datetime](https://www.mathworks.com/help/matlab/ref/datetime.html), [caldiff](https://www.mathworks.com/help/matlab/ref/datetime.caldiff.html), and [split](https://www.mathworks.com/help/matlab/ref/split.html). These were useful for such a task.

### Matlab Code for Accessing Data and Constructing Time Vector

```matlab
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Notes:
% topoChico4.m will read through Shen's data set
% and write to 3D array in .nc format
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Writing Time Vector 
% read colab data and study time vector 
% to see values after 1880-01-01 (or w/e time is)
t_0 = 1950; t_f = 2012;
nt = 12*(t_f-t_0+1);

t = [];
t_new = t_0;
cnt = 1;
for jj=1:(t_f-t_0+1)
    for kk=1:12
        t = [t, datetime(t_new,kk,1)];
    end
    t_new = t_new+1;
end
time_calendarDate = caldiff(t,'days')';
% days after 1950-01-01
time_temp = split(time_calendarDate,'days');
time = zeros(nt,1);
time(1) = 0;
sum = 0;
for jj=1:nt-1
    sum = sum + time_temp(jj);
   time(jj+1) = sum;
end
%% Writing Temperature
% lon (nx), lat (ny), time (nt)
nx = 360;
ny = 180;
nt = nt;
% 3D array of zeros 
oceanTemp = zeros(nx,ny,nt);
months = ["Jan", "Feb", "Mar", "Apr", ...
    "May", "Jun", "Jul", "Aug", "Sept", ...
    "Oct", "Nov", "Dec"];
year_Vec = t_0:t_f;
depth_selection = 100;
% takes me to correct folder 
temp1_dir = strcat('C:\Users\Jesus Perez Cuarenta\Documents\MATLAB\NOAA Fellowship\Data_SSS_Ocean\Data_SSS_Ocean\Data\2D_Reconstructions_5m-2000m\',...
    num2str(depth_selection),'m\');
% translates selected depth to string
temp2_dir = strcat(num2str(depth_selection),'m');
% need string here 
temp3_dir = '_Reconstructed_Temp_Anomaly_';
% need MonthYr Here 

desiredDir = @(tot_count,current_month,year_count) strcat(temp1_dir,...
    temp2_dir,num2str(tot_count),temp3_dir,months(current_month),...
    num2str(year_Vec(year_count)),'.mat');    
cnt=1;
yr_cnt = 1;
for kk=1:length(year_Vec)
    for jj=1:12
        current_mat = load(desiredDir(cnt,jj,yr_cnt));
        % tranpose so orientation matches GODAS set 
        data = current_mat.M';
        oceanTemp(:,:,cnt) = data;
        cnt=cnt+1;
    end
    yr_cnt=yr_cnt+1;
end
%% Sanity Check
for jj=1:756
    imagesc(rot90(oceanTemp(:,:,jj)));
    colorbar;
    caxis([-4,4])
    pause(0.1)
end
%% Write nc file
% Check useful colab variables
colab_ncfile = 'C:\Users\Jesus Perez Cuarenta\Documents\MATLAB\NOAA Fellowship\sst.mon.mean.trefadj.anom.1880to2018.nc';
colab_sst = ncread(colab_ncfile,'sst');
colab_time = ncread(colab_ncfile,'time');

% "steal" lon and lat 
lon = ncread(colab_ncfile,'lon');
lat = -ncread(colab_ncfile,'lat');

nccreate(my_ncfile,'time','Dimensions',{'time',1,Inf});
ncwrite(my_ncfile,'time',time);
ncwriteatt(my_ncfile,'time','standard_name','time');
ncwriteatt(my_ncfile,'time','long_name','Time');
ncwriteatt(my_ncfile,'time','units','days since 1950-1-1 00:00:00');
ncwriteatt(my_ncfile,'time','calendar','standard');
ncwriteatt(my_ncfile,'time','axis','T');

nccreate(my_ncfile,'lon','Dimensions',{'lon',1,nx});
ncwrite(my_ncfile,'lon',lon);
ncwriteatt(my_ncfile,'lon','standard_name','longitude');
ncwriteatt(my_ncfile,'lon','long_name','Longitude');
ncwriteatt(my_ncfile,'lon','units','degrees_east');
ncwriteatt(my_ncfile,'lon','axis','X');

nccreate(my_ncfile,'lat','Dimensions',{'lat',1,ny});
ncwrite(my_ncfile,'lat',lat);
ncwriteatt(my_ncfile,'lat','standard_name','latitude');
ncwriteatt(my_ncfile,'lat','long_name','Latitude');
ncwriteatt(my_ncfile,'lat','units','degrees_north');
ncwriteatt(my_ncfile,'lat','axis','Y');

my_ncfile = 'shenData.nc';
nccreate(my_ncfile,'deepTemp','Dimensions',{'lon','lat','time'},'Datatype','single') ;
ncwrite(my_ncfile,'deepTemp',oceanTemp);

```

### Results
The results are not promising compared to Week 1. I look forward to discussing with Dr. Shen next Monday and figure out if I overlooked something in my code. 


```python
# Modified code originally provided by Ankur Mahesh from ClimateAi.
def load_enso_indices():
  """
  Reads in the txt data file to output a pandas Series of ENSO vals

  outputs
  -------

    pd.Series : monthly ENSO values starting from 1950-01-01
  """

  with open(anomaly_path) as f:
    line = f.readline()
    enso_vals = []
    while line:
        yearly_enso_vals = map(float, line.split()[1:]) 
        enso_vals.extend(yearly_enso_vals)
        line = f.readline()

  enso_vals = pd.Series(enso_vals)
  enso_vals.index = pd.date_range('1950-01-01',freq='MS',
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
#  figName_png = str(depth)+'m_Predictions.png'
#  figName_pdf = str(depth)+'m_Predictions.pdf'
  figName_png = 'Shen_100m_Predictions.png'
  figName_pdf = 'Shen_100m_Predictions.pdf'

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
depth_arr = 100

# for ii in range(len(depth_arr)):
depth_sel = depth_arr
# Read data from Drive
godas_path = "/content/gdrive/MyDrive/ColabNotebooks/Yearly_NC_Files_Plus_Anomalies/shenData.nc"
anomaly_path = "/content/gdrive/MyDrive/ColabNotebooks/Yearly_NC_Files_Plus_Anomalies/Shen_deepTemp_anomalies_100m.txt"
# Select lead time 
month_lead = 1
# Training and validation data sets
X_train, y_train = assemble_basic_predictors_predictands('1980-01-01','1995-12-31', lead_time=month_lead)
X_val, y_val = assemble_basic_predictors_predictands('1997-01-01','2006-12-31', lead_time=month_lead)

regr = sklearn.linear_model.LinearRegression()
regr.fit(X_train,y_train)
predictions = regr.predict(X_val)
corr, _ = scipy.stats.pearsonr(predictions, y_val)

# Save figure
plot_nino_time_series(y_val, predictions, depth_sel,
    'Predicted and True Ocean Temperature Anomalies at'+' '+str(month_lead)+' '+'Month Lead Time. \n Corr: {:.2f}'.format(corr))
```


<figure>
<center>
<img src='https://drive.google.com/uc?export=view&id=1GolXnmBTgUvU3Akc2C-FmKsy7FjKvtLY' />
<figcaption> Figure 11: Initial simulation with Dr. Shen's data set. Most likely there are bugs in my code. </figcaption></center>
</figure>

