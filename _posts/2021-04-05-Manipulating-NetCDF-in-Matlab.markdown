---
layout: post
title:  "Manipulating NetCDF in Matlab"
author: "Jesus Perez Cuarenta"
categories: journal
tags: [documentation]

---
For our first post we will explore how to read and create .nc 
files within Matlab. The files we will be using are hosted 
at the [National Oceanic and Atmospheric Administration (NOAA) Physical Sciences Laboratory](https://www.psl.noaa.gov/data/gridded/data.godas.html) 
website. We will also use the [COBE SST data set](ftp://ftp.cdc.noaa.gov/Datasets/COBE/sst.mon.mean.nc). 
The latter file can be downloaded manually. I saved the file as *cobeData.nc*, which can be 
modified -- just make sure to change the code accordingly. 
 
Originally, my task involved rewriting the GODAS data set as a single .nc 
file to match the format from the COBE set. Hence, the code in this document
involves reading .nc files from distinct data sets. 

**Note:** The following Matlab code requires *convtemp* from the Aerospace toolbox 
for easily converting Kelvin to Celsius. If you wish to keep Kelvin units, or apply the conversion yourself feel free to omit the line where *convtemp* makes an appearance. 

First, we download the yearly GODAS .nc files automatically. These files are placed 
in the current Matlab folder. The location in question can be obtained easily.
Double check the current folder contains the COBE data set. 
{% highlight matlab %}
pwd
txtNC = pwd;
{% endhighlight %}

Next we download the GODAS data set. This step takes a couple of minutes. 
{% highlight matlab %}
ftobj = ftp('ftp2.psl.noaa.gov');
cd(ftobj,'Datasets/godas');
jj = 1980;
txtLeft = 'pottmp.';
txtRight = '.nc';
while jj < 2020
    txtTemp = num2str(jj);
    InputTxt = strcat(txtLeft,txtTemp,txtRight);
    mget(ftobj,InputTxt);
    jj=jj+1;
end
close(ftobj)
{% endhighlight %}

Now we need an efficient way of obtaining an .nc file for a given year. 
We also declare the name for the final .nc file we will create. 
{% highlight matlab %}
ncFunc = @(kk) strcat(txtNC,'\pottmp.',num2str(kk),'.nc');
Yr_vec = 1980:1:2020;

% Declare final output file title as string 
my_ncfile = 'godasData.nc';
{% endhighlight matlab %}

Reading .nc files in Matlab is simple. We just need the right string title. 
{% highlight matlab %}
% We take the 1980 GODAS .nc file to understand how the set is structured. 
godas_ncfile = ncFunc(1980);
% Now for the COBE .nc file. 
cobe_ncfile = strcat(txtNC,'\cobeData.nc');
{% endhighlight %}

Once you have these declared these strings, I highly recommend running *ncdisp(godas_ncfile)* and
*ncdisp(cobe_ncfile)* to understand what information is in each one. 
We are now ready to obtain the longitude, latitude, depth, and time values from both data sets. 

{% highlight matlab %}
% Read 'lon', 'lat', 'sst' and 'time' variables from COBE file
% as arrays. 
cobe_lon = ncread(cobe_ncfile,'lon');
cobe_lat = ncread(cobe_ncfile,'lat');
cobe_sst = ncread(cobe_ncfile,'sst');
time_temp = ncread(cobe_ncfile,'time');

% Read 'lon', 'lat', 'time', 'level' variables from GODAS file 
% Read longitude 
lon = ncread(godas_ncfile,'lon');
% Read latitude 
lat = ncread(godas_ncfile,'lat');
% Read level 
% If you want to save all depths
% change depthIndices accordingly
level_temp = ncread(godas_ncfile,'level');
depthIndices = 11:21;
level = level_temp(depthIndices);
% Read time 
% The value of 1201 for time_temp is particular to my task so change accordingly
time = time_temp(1201:end);

% Knowing the length of these arrays will soon be useful 
nx = length(lon);
ny = length(lat);
nt = length(time);
nd = length(depthIndices);
{% endhighlight %}

We are now ready to define the variables our .nc file will encode.
This is possible with the *nccreate* command. The command *ncwrite* writes 
the data (array) under the selected variable (string). Last, *ncwriteatt* is optional but
useful if you will be sharing your .nc file so others can understand 
the structure. See Matlab documentation for a more detailed explanation.

{% highlight matlab %}
% Time 
nccreate(my_ncfile,'time','Dimensions',{'time',1,Inf});
ncwrite(my_ncfile,'time',time);
ncwriteatt(my_ncfile,'time','standard_name','time');
ncwriteatt(my_ncfile,'time','long_name','Time');
ncwriteatt(my_ncfile,'time','units','days since 1891-1-1 00:00:00');
ncwriteatt(my_ncfile,'time','calendar','standard');
ncwriteatt(my_ncfile,'time','axis','T');

% Longitude
nccreate(my_ncfile,'lon','Dimensions',{'lon',1,nx});
ncwrite(my_ncfile,'lon',lon);
ncwriteatt(my_ncfile,'lon','standard_name','longitude');
ncwriteatt(my_ncfile,'lon','long_name','Longitude');
ncwriteatt(my_ncfile,'lon','units','degrees_east');
ncwriteatt(my_ncfile,'lon','axis','X');

% Latitude 
nccreate(my_ncfile,'lat','Dimensions',{'lat',1,ny});
ncwrite(my_ncfile,'lat',lat);
ncwriteatt(my_ncfile,'lat','standard_name','latitude');
ncwriteatt(my_ncfile,'lat','long_name','Latitude');
ncwriteatt(my_ncfile,'lat','units','degrees_north');
ncwriteatt(my_ncfile,'lat','axis','Y');

% Depth 
nccreate(my_ncfile,'level','Dimensions',{'level',1,nd});
ncwrite(my_ncfile,'level',level);
{% endhighlight %}

Now that we know the length of each dimension we proceed to 
define a 4D array of zeros and loop through all GODAS yearly files. 

{% highlight matlab %}
% 4D array of zeros 
kel_seaTemp = zeros(nx,ny,nd,nt);

% For convenience regarding entries in the time dimension 
tempVec = @(jj) [1:12]+12*(jj-1);

% Loop through all depths 
for jj=1:nd
	% Loop through all years (excluding last two)
    for kk = 1:length(Yr_vec)-2
        yr = Yr_vec(kk);
        % read yearly GODAS data 
        godas_ncfile = ncFunc(yr);
        
        % read yearly GODAS temperature data 
        pottmp_current = ncread(godas_ncfile,'pottmp');
        
        % Store 3D array (Nx,Ny,Nt) corresponding to yr
        seaTemp_current = pottmp_current(:,:,depthIndices(jj),:);
        tempvec = tempVec(kk);
        kel_seaTemp(:,:,jj,tempvec) = squeeze(seaTemp_current);
    end
end

% Change missing values to NaN 
kel_seaTemp(kel_seaTemp == -9.969209968386869e+36) = NaN;


% Adjust temperature from Kelvin to Celsius.
cel_seaTemp = convtemp(kel_seaTemp,'K','C');
{% endhighlight %}

To conclude, we write the *cel_seaTemp* array in our .nc file. 

{% highlight matlab %}
nccreate(my_ncfile,'deepTemp','Dimensions',{'lon','lat','level','time'},'Datatype','single') ;
ncwrite(my_ncfile,'deepTemp',cel_seaTemp);
ncwriteatt(my_ncfile,'deepTemp','long_name','Monthly Means of Ocean Temperature');
ncwriteatt(my_ncfile,'deepTemp','units','degC');
{% endhighlight %}

That's it! To view your final result you may write 
{% highlight matlab %}
ncdisp(my_ncfile)
{% endhighlight %}
in the command window. 
