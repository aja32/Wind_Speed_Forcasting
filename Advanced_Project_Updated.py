# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 17:29:01 2021

@author: THINKPAD
"""



import os
os.environ["PROJ_LIB"] = r'C:\Users\THINKPAD\anaconda3\Library\share\basemap'

from mpl_toolkits.basemap import Basemap

import pandas as pd
import matplotlib.pyplot as plt
from pandas import Timestamp
import seaborn as sns
import matplotlib
import numpy as np
import math

t1 = pd.read_csv("1.csv")


#################   Cleaning Data

t1.columns = t1.columns.str.lstrip()
t1.drop(columns = ["rated power output at 100m (MW)","CorrectedScore"],inplace = True)

#print(t1[t1['100m wind speed (m/s)'] == ''].sum())
print(pd.isna(t1).sum())

t1 = t1.rename(columns = {"Date(YYYY-MM-DD hh:mm:ss)":"Date","100m wind speed (m/s)":"Wind_Speed_(m/s)","SCORE-lite power output at 100m (MW)":"Power_Output_(MW)"})
#print(t1)

#################################################


## Wind speed vs Time

plt.plot(t1.iloc[0:5000,1])
#plt.plot(t1.iloc[0:5000:,2])
plt.title("100 m wind speed TurbineID1 2006")
plt.xlabel("Time [10 min] From 1/1/2006  00:00:00 till 2/4/2006  17:00:00")
plt.ylabel("Wind speed at distance 100m [m/s]")
plt.show()


## Power vs Time

plt.figure()
#plt.plot(t1.iloc[0:5000,1])
plt.plot(t1.iloc[0:5000:,2])
plt.title("rated power output at 100m")
plt.xlabel("Time [10 min] From 1/1/2006  00:00:00 till 2/4/2006  17:00:00")
plt.ylabel("rated power output in MW")
plt.show()


## Wind Speed + Power vs Time

plt.figure()
plt.plot(t1.iloc[0:5000,1],label= "Wind Speed")
plt.plot(t1.iloc[0:5000:,2], label = "Rated Power")
plt.legend()
plt.title("rated power output and Wind Speed t at 100m")
plt.xlabel("Time [10 min], From 1/1/2006  00:00:00 till 2/4/2006  17:00:00")
plt.ylabel("rated power output in MW")
plt.show()


######################################################################
###########3 check for different seasons  #####################
###################################33#########3##############################
## Wind Speed vs Time in Summer:

# Data Preparation
plt.figure()
t1_summer = t1.iloc[24626:37874,1:3]
t1_summer = t1_summer.reset_index()
t1_summer.drop(columns = "index",inplace = True)
print(t1_summer)


#################################
plt.plot(t1_summer.iloc[:,0])
plt.title("100 m wind speed TurbineID1 2006 in Summer")
plt.xlabel("Time [10 min, From 6/21/2006  00:00:00 Till 9/21/2006  00:00:00")
plt.ylabel("Wind Speed at distance 100 [m/s]")
plt.show()


## Power vs Time in Summer

plt.plot(t1_summer.iloc[:,1])
plt.title("rated power output at 100m in Summer")
plt.xlabel("Time [10 min],From 6/21/2006  00:00:00 Till 9/21/2006  00:00:00")
plt.ylabel("rated power output in MW")
plt.show()


##################################################
####

## Wind Speed vs Time in Fall:

    
# Data Preparation
plt.figure()
t1_fall = t1.iloc[37874:50978,1:3]
t1_fall = t1_fall.reset_index()
t1_fall.drop(columns = "index",inplace = True)
print(t1_fall)


plt.figure()
plt.plot(t1_fall.iloc[:,0])
plt.title("100 m wind speed TurbineID1 2006 in Fall")
plt.xlabel("Time [10 min], From 9/21/2006  00:00:00 till 12/21/2006  00:00:00")
plt.ylabel("Wind Speed at distance 100 [m/s]")
plt.show()


## Power vs Time in Fall

plt.figure()
plt.plot(t1_fall.iloc[:,1])
plt.title("rated power output at 100m in Fall")
plt.xlabel("Time [10 min], From 9/21/2006  00:00:00 till 12/21/2006  00:00:00")
plt.ylabel("rated power output in MW")
plt.show()




##########################################
##########################################
############################################
############################################


####################################################
###################################################3

## Histogram for Wind Speed

plt.figure()
plt.hist(t1.iloc[:,1],edgecolor = "black",color = "blue",bins = 35)
plt.title("Histogram shows the wind speed in 2006 of TurbineID1")
plt.xlabel("Wind Speed (m/s)")
plt.ylabel("Count")
plt.show()


## Histogram for Power

plt.figure()
plt.hist(t1.iloc[:,2],edgecolor = "black",color = "blue",bins = 35)
plt.title("Histogram shows the Power in 2006 of TurbineID1")
plt.xlabel("Power values (MW)")
plt.ylabel("Count")
plt.show()



#########################################################
############################################################


## Histogram for Wind Speed in Summer

plt.figure()
plt.hist(t1_summer.iloc[:,0],edgecolor = "black",color = "blue",bins = 35)
plt.title("Histogram shows the wind speed in 2006 of TurbineID1 in Summer")
plt.xlabel("Wind Speed (m/s)")
plt.ylabel("Count")
plt.show()


## Histogram for Power

plt.figure()
plt.hist(t1_summer.iloc[:,1],edgecolor = "black",color = "blue",bins = 35)
plt.title("Histogram shows the Power in 2006 of TurbineID1 in Summer")
plt.xlabel("Power values (MW)")
plt.ylabel("Count")
plt.show()


################################################################
############################################################

## Histogram for Wind Power in fall

plt.figure()
plt.hist(t1_fall.iloc[:,0],edgecolor = "black",color = "blue",bins = 35)
plt.title("Histogram shows the Power in 2006 of TurbineID1 in Fall")
plt.xlabel("Wind Speed (m/s)")
plt.ylabel("Count")
plt.show()


## Histogram for Power in Fall

plt.figure()
plt.hist(t1_fall.iloc[:,1],edgecolor = "black",color = "blue",bins = 35)
plt.title("Histogram shows the Power in 2006 of TurbineID1 in Fall")
plt.xlabel("Power values (MW)")
plt.ylabel("Count")
plt.show()



####################################################################
########################################################################


################## Note some missing values for winter(Data not provided)

## Histogram for Wind Power in Winter

plt.figure()
plt.hist(t1.iloc[0:11520,1],edgecolor = "black",color = "blue",bins = 35)
plt.title("Histogram shows the Power in 2006 of TurbineID1 in Winter")
plt.xlabel("Wind Speed (m/s)")
plt.ylabel("Count")
plt.show()


## Histogram for Power in Fall

plt.figure()
plt.hist(t1.iloc[0:11520,2],edgecolor = "black",color = "blue",bins = 35)
plt.title("Histogram shows the Power in 2006 of TurbineID1 in Winter")
plt.xlabel("Power values (MW)")
plt.ylabel("Count")
plt.show()




################################################################
#################################################################
################################################################3


### Find the average of Wind Speed and Power for whole year and for each season(For TurbineID1):

## For Wind Speed

# Winter
x = t1.iloc[0:11520,1].mean()
print("The average Wind Speed in Winter for TurbineID1 is:{} m/s".format(x))

# Spring
x = t1.iloc[11521:24625,1].mean()
print("The average Wind Speed in Spring for TurbineID1 is:{} m/s".format(x))

# Summer
x = t1.iloc[24626:37873,1].mean()
print("The average Wind Speed in Summer for TurbineID1 is:{} m/s".format(x))

# Fall
x = t1.iloc[37874:50977,1].mean()
print("The average Wind Speed in Fall for TurbineID1 is:{} m/s".format(x))

# Whole year
x = t1.iloc[:,1].mean()
print("The average Wind Speed of TurbineID1 in 2006 is:{} m/s".format(x))




## For Wind Power

# Winter
x = t1.iloc[0:11520,2].mean()
print("The average Wind Power in Winter for TurbineID1 is:{} MW".format(x))

# Spring
x = t1.iloc[11521:24625,2].mean()
print("The average Wind Power in Spring for TurbineID1 is:{} MW".format(x))

# Summer
x = t1.iloc[24626:37873,2].mean()
print("The average Wind Power in Summer for TurbineID1 is:{} MW".format(x))

# Fall
x = t1.iloc[37874:50977,2].mean()
print("The average Wind Power in Fall for TurbineID1 is:{} MW".format(x))

# Whole year
x = t1.iloc[:,2].mean()
print("The average Wind Power of TurbineID1 in 2006 is:{} MW".format(x))



###################################################
####################################################
#####################################################

# Preparing data to be used for heatmaps

t1["Days"] = t1.iloc[:,0].apply(lambda x: Timestamp(x).day)
print(t1["Days"])

t1["Month"] = t1.iloc[:,0].apply(lambda x: Timestamp(x).strftime("%B"))

print(t1)


New_Data_Frame = t1.pivot_table(index = "Month",values = "Wind_Speed_(m/s)",aggfunc = "mean",columns = "Days")
print(New_Data_Frame)

### In order to have our pivot_data orderd by month, we will reorder our index

New_Index = ["January", "February", "March", "April","May", "June", "July", "August", "September", "October", "November", "December"]

New_Data_Frame = New_Data_Frame.reindex(New_Index)
print(New_Data_Frame)

###########################################################################
###########################################################################
##########################################################################
##########################################################################


print("Trying to work on dashed curve")

## Choose 3 months to reflect the wind speed average/Day in different Seasons
zz = New_Data_Frame.loc[["January","July","October"],:]
Month_Names = zz.index
y_January_df = zz.iloc[0,:]    ## extracting Wind Speed vsdays of January
y_January = y_January_df.values.tolist()
y_July = zz.iloc[1,:].values.tolist()
y_October = zz.iloc[2,:].values.tolist()
x_axis = y_January_df.index   ### Days to be the x-axis


## Draw plot that shows Wind speed average/Day for 3 month

plt.figure()
line_January = plt.plot(x_axis,y_January,color = "black",linestyle = "dashed",linewidth = 2,
         marker='o', markerfacecolor='blue', markersize=5, label ="January")
line_July = plt.plot(x_axis,y_July,color = "black",linestyle = "solid",linewidth = 2,
         marker='o', markerfacecolor='red', markersize=5, label ="August")
line_October = plt.plot(x_axis,y_October,color = "black",linestyle = "dotted",linewidth = 2,
         marker='o', markerfacecolor='green', markersize=5, label ="October")

plt.legend()

# naming the x axis
plt.xlabel('Days')
# naming the y axis
plt.ylabel('Wind Speed/Day [m/s]')
  
# giving a title to my graph
plt.title('The Wind Speed average/Day for 3 different Month in 2006')
  
# function to show the plot
plt.show()

print(zz)





####################################################################
##########################################################################
###################  For Power Curves (3 month)   ########################
##########################################################################

New_Data_Frame_power = t1.pivot_table(index = "Month",values = "Power_Output_(MW)",aggfunc = "mean",columns = "Days")
print(New_Data_Frame_power)

### In order to have our pivot_data orderd by month, we will reorder our index

New_Index = ["January", "February", "March", "April","May", "June", "July", "August", "September", "October", "November", "December"]

New_Data_Frame_power = New_Data_Frame_power.reindex(New_Index)
print(New_Data_Frame_power)

###########################################################################


print("Trying to work on dashed curve")

## Choose 3 months to reflect the Power average/Day in different Seasons
zz = New_Data_Frame_power.loc[["January","July","October"],:]
Month_Names = zz.index
y_January_df = zz.iloc[0,:]    ## extracting Power vsdays of January
y_January = y_January_df.values.tolist()
y_July = zz.iloc[1,:].values.tolist()
y_October = zz.iloc[2,:].values.tolist()
x_axis = y_January_df.index   ### Days to be the x-axis


## Draw plot that shows Power average/Day for 3 month

plt.figure()
line_January = plt.plot(x_axis,y_January,color = "black",linestyle = "dashed",linewidth = 2,
         marker='o', markerfacecolor='blue', markersize=5, label ="January")
line_July = plt.plot(x_axis,y_July,color = "black",linestyle = "solid",linewidth = 2,
         marker='o', markerfacecolor='red', markersize=5, label ="August")
line_October = plt.plot(x_axis,y_October,color = "black",linestyle = "dotted",linewidth = 2,
         marker='o', markerfacecolor='green', markersize=5, label ="October")

plt.legend()

# naming the x axis
plt.xlabel('Days')
# naming the y axis
plt.ylabel('SCORE-lite power output at 100m (MW)')
  
# giving a title to my graph
plt.title('The Power for 3 different Month in 2006')
  
# function to show the plot
plt.show()

##############################################################################
###################################################################################
################################################################

### Month vs Wind Speed in 2006
## Add the average wind speed for each month (For TurbineID1):

New_Data_Frame["Average_Wind_Speed"] = New_Data_Frame.mean(axis = 1)
print(New_Data_Frame)

x_Axis = New_Data_Frame.index
print(x_Axis)
print(x_Axis.values.tolist())

## Put wind speed values in y-axis
y_Axis = New_Data_Frame.iloc[:,31].values.tolist()  
print(y_Axis)

## Plotting figure

plt.figure()
plt.plot(x_Axis,y_Axis,color = "black",linestyle = "solid",linewidth = 2,
         marker='o', markerfacecolor='blue', markersize=5)
# Rotate s-axis labels to make it clearer
plt.xticks(rotation=45)
# naming the x axis
plt.xlabel('Month')
# naming the y axis
plt.ylabel('Wind Speed/Day [m/s]')
plt.title("Wind Speed average per Month in 2006")
plt.show()




###################################################################
########################################################################
########################################################################

## Draw Heatmap
sns.heatmap(New_Data_Frame, annot=False, cmap="coolwarm")
plt.title("Heatmap shows average Wind speed  of TurbineID1 in 2006")
plt.show()

########################################################################
#################################################################333#
## Days vs Hours

t1["Hours"] = t1.iloc[:,0].apply(lambda x: Timestamp(x).strftime("%H"))

New_Data_Frame_plus = t1.pivot_table(index = "Hours",values = "Wind_Speed_(m/s)",columns = "Days",aggfunc = "mean")
New_Data_Frame_plus["Average_hours"] = New_Data_Frame_plus.mean(axis = 1)
sns.heatmap(New_Data_Frame_plus, annot=False, cmap="coolwarm")
plt.title("Heatmap shows average Wind speed  of TurbineID1 in 2006--Hours vs days")
plt.show()


#################################################################333#
## Hours vs Minutes

t1["Minutes"] = t1.iloc[:,0].apply(lambda x: Timestamp(x).strftime("%M"))

New_Data_Frame_plus_Minutes = t1.pivot_table(index = "Minutes",values = "Wind_Speed_(m/s)",columns = "Hours",aggfunc = "mean")
#New_Data_Frame_plus_Minutes["Av_Minutes"] = New_Data_Frame_plus_Minutes.mean(axis = 1)
sns.heatmap(New_Data_Frame_plus_Minutes, annot=False, cmap="coolwarm")
plt.title("Heatmap shows average Wind speed  of TurbineID1 in 2006--Minutes vs Hours")
plt.show()


print(New_Data_Frame_plus_Minutes)




####################################################################3
#####################################################################

# HeatMap for Power ( for each month)

Data_Frame_New_Power = t1.pivot_table(values = "Power_Output_(MW)",index = "Month",columns = "Days", aggfunc = "mean")
Data_Frame_New_Power["Av_Power"] = Data_Frame_New_Power.mean(axis = 1)
Data_Frame_New_Power = Data_Frame_New_Power.reindex(New_Index)
sns.heatmap(Data_Frame_New_Power,annot=False, cmap="coolwarm")
plt.title("Heatmap shows average Wind Power  of TurbineID1 in 2006")
plt.show()

#################


# Try to check why the avrge are not appearing on heat maps(Values are found, 
# but no names)

#############################
##############################################
########################################################
####################################################

## 2nd Data " SITE_META "



site_meta_data = pd.read_csv("site_meta.csv")
site_meta_data.columns = site_meta_data.columns.str.lstrip()
site_meta_data = site_meta_data[["SiteID","Latitude","Longitude","Power Density [W/m2]","SCORE-lite Capacity Factor [%]","Wind Speed [m/s]","State Code","Model Elevation [m]"]]
print(site_meta_data)

#change name of columns since we always prefer them without any spaces.

site_meta_data.columns = ["id","lats","lons","power_density","power_capacity","wind_speed", "state","altitude"]
print(site_meta_data)

# setting parameters for title and axes
#font = {'family' : 'verdana','size' : 16}
#plt.rc(font, **font)



# How much to zoom from coordinates (in degrees)
zoom_scale = 3
# Setup the bounding box for the zoom and bounds of the map
bbox = [min(site_meta_data.lats)-zoom_scale,max(site_meta_data.lats)+zoom_scale,\
min(site_meta_data.lons)-zoom_scale,max(site_meta_data.lons)+zoom_scale]
    
fig, ax = plt.subplots(figsize=(12,7))
plt.title("Position of Wind turbines from NREL data with their average Wind speed")



#####
# Define the projection, scale, the corners of the map, and the resolution.
m = Basemap(projection='merc',llcrnrlat=bbox[0],urcrnrlat=bbox[1],\
llcrnrlon=bbox[2],urcrnrlon=bbox[3],lat_ts=10,resolution='i')
# Draw coastlines and fill continents and water with color
m.drawcoastlines()
m.fillcontinents(color='#CCCCCC',lake_color='lightblue')




# labels = [left,right,top,bottom]
# draw parallels, meridians, and color boundaries
m.drawparallels(np.arange(bbox[0],bbox[1],(bbox[1]-bbox[0])/5),labels=[1,0,0,0])
m.drawmeridians(np.arange(bbox[2],bbox[3],(bbox[3]-bbox[2])/5),labels=[0,0,0,1],rotation=15)
m.drawmapboundary(fill_color="lightblue")


##############
# format colors for wind speed range
ws_min = min(site_meta_data.wind_speed)
ws_max = max(site_meta_data.wind_speed)
cmap = plt.get_cmap('gist_earth')  ###choosing color used in maps(array)
normalize = matplotlib.colors.Normalize(vmin=ws_min, vmax=ws_max)

# plot elevations with different colors using the numpy interpolation mapping tool
# the range [20,170] can be changed to create different colors and ranges

for ii in range(0,len(site_meta_data.wind_speed)):
    x,y = m(site_meta_data.lons[ii],site_meta_data.lats[ii])
    color_interp = np.interp(site_meta_data.wind_speed[ii],[ws_min,ws_max],[20,170])
    plt.plot(x,y,3,marker='o',color=cmap(int(color_interp)))
    
    
cax, _ = matplotlib.colorbar.make_axes(ax)  ## creating a color bar axes as part of our plot
cbar = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap,norm = normalize,label='Average Wind Speed')

plt.show()


#######################################################
######################################################




## For elevation

# How much to zoom from coordinates (in degrees)
zoom_scale = 3   

bbox = [min(site_meta_data.lats)-zoom_scale,max(site_meta_data.lats)+zoom_scale,min(site_meta_data.lons)-zoom_scale,max(site_meta_data.lons)+zoom_scale] 

fig, ax =  plt.subplots(figsize=(12,7))
plt.title("Position of Wind turbines from NREL data with their average Elevation")

# Define the projection, scale, the corners of the map, and the resolution.
m = Basemap(projection='merc',llcrnrlat=bbox[0],urcrnrlat=bbox[1],\
llcrnrlon=bbox[2],urcrnrlon=bbox[3],lat_ts=10,resolution='i')
    
m.drawcoastlines()
m.fillcontinents(color='#CCCCCC',lake_color='lightblue')

# labels = [left,right,top,bottom]
# draw parallels, meridians, and color boundaries
m.drawparallels(np.arange(bbox[0],bbox[1],(bbox[1]-bbox[0])/5),labels=[1,0,0,0])
m.drawmeridians(np.arange(bbox[2],bbox[3],(bbox[3]-bbox[2])/5),labels=[0,0,0,1],rotation = 15)
m.drawmapboundary(fill_color="lightblue")

##############
# format colors for wind speed range
elev_min = min(site_meta_data.altitude)
elev_max = max(site_meta_data.altitude)
cmap = plt.get_cmap('gist_earth')  ###choosing color used in maps(array)
normalize = matplotlib.colors.Normalize(vmin=elev_min, vmax=elev_max)

for ii in range(0,len(site_meta_data.altitude)):
    x,y = m(site_meta_data.lons[ii],site_meta_data.lats[ii])
    color_interp = np.interp(site_meta_data.altitude[ii],[elev_min,elev_max],[50,200])
    plt.plot(x,y,3,marker='o',color=cmap(int(color_interp)))
    
    
cax, _ = matplotlib.colorbar.make_axes(ax)  ## creating a color bar axes as part of our plot
cbar = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap,norm = normalize,label='Elevation')
plt.show()



###################################################################################3
#################################################################################3


site_meta_data_Corr = pd.read_csv("site_meta.csv")
site_meta_data_Corr = site_meta_data_Corr.iloc[:,0:8]
print(site_meta_data_Corr)
print(site_meta_data_Corr.corr().shape)
sns.heatmap(site_meta_data_Corr.corr(), annot=True)
plt.show()



#######################################################################
######################################################################
#########3            Modeling   #####################################
#######################################################################
######################################################################


################################################################
#################################################################
######################  AG  ####################################
######################################################################
######################################################################

# create and evaluate a static autoregressive model
#from pandas import read_csv
#from matplotlib import pyplot
from statsmodels.tsa.ar_model import AutoReg
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.model_selection import train_test_split
from statsmodels.tsa.stattools import adfuller


# load dataset
series = pd.read_csv('1.csv', header=0, index_col=0, parse_dates=True, squeeze=True)

## Cant check if the data stationary or not, so we will check using adfuller
series[" SCORE-lite power output at 100m (MW)"].plot()

#Use adfuller to check for stationarity
series_test = adfuller(series[" SCORE-lite power output at 100m (MW)"],autolag="AIC")

print("1. ADF : " ,series_test[0])
print("2. P-Value : " ,series_test[1])
print("3. Num of Lags : " ,series_test[2])
print("4. # of Observations : " ,series_test[3])



## since P-value is less than 0.5 -->> stationary data 
# Since we reject the hypothesis of being nonstationary
# because the probability of getting a p-value as low as that 
# by mere luck (random chance) is very unlikely

plt.acorr(series[" SCORE-lite power output at 100m (MW)"],maxlags= 30000)
plt.show()


## check for the lags
# usually we use partial cf for deciding the correlation, and not the autocorrelation.
# since auto depends on several indirect effects.
#Pcaf measures the direct effect of particular time period of time with the current time period.
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf

pacf = plot_pacf(series[" SCORE-lite power output at 100m (MW)"],lags = 20)
acf = plot_acf(series[" SCORE-lite power output at 100m (MW)"],lags = 20)


train, test = train_test_split(series, train_size = 0.8)
print("The train set is :")
print(train)
print("....")

print("The test set is :")
print(test)
print(test.size)

train = train[" SCORE-lite power output at 100m (MW)"]
test = test[" SCORE-lite power output at 100m (MW)"]
print(train)

train = train.values
test = test.values
print(train)
print("bekh")
print(test)




# train autoregression
model = AutoReg(train, lags=3)
model_fit = model.fit()
print('Coefficients: %s' % model_fit.params)

### ########Check for overfitting :
predictions = model_fit.predict(start=0, end=len(train), dynamic=False)
# plot results
plt.figure()
plt.plot(train)
plt.plot(predictions, color='red')
plt.show()
#################################  
# not overfiting
#############################3
# make predictions
predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)

# check the difference betweeen predictions and real test dataset
for i in range(len(predictions)):
	print('predicted=%f, expected=%f' % (predictions[i], test[i]))

rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)
# plot results
plt.figure()
plt.plot(test)
plt.plot(predictions, color='red')
plt.show()






################################################
################################################################
############################# ARIMA ####################################
####################################################################
######################################################################

series = series.rename(columns = {"100m wind speed (m/s)":"WindSpeed"})
ARIMA_Data = series.iloc[0:1000]




from statsmodels.tsa.stattools import adfuller
def check_stationarity(timeseries):
    result = adfuller(timeseries,autolag='AIC')
    dfoutput = pd.Series(result[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    print('The test statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('%s: %.3f' % (key, value))
        
        
check_stationarity(ARIMA_Data.WindSpeed)




#without differentation
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(ARIMA_Data.WindSpeed,lags=10)

#with 1 differentiation.
plot_acf(ARIMA_Data.WindSpeed.diff().dropna())


## To check pacf after differentiation. -  find p
from statsmodels.graphics.tsaplots import plot_pacf
plt.figure(figsize=(10,2))
plot_pacf(ARIMA_Data.WindSpeed.diff().dropna())






from statsmodels.tsa.arima_model import ARIMA
mod = ARIMA(ARIMA_Data.WindSpeed,order=(11,0,4))
results = mod.fit()


### checking 


residuals = pd.DataFrame(results.resid)
residuals.plot()
residuals.plot(kind='kde')
residuals.describe()
plot_pacf(residuals)
####################
####################
#####################
#########################
###############################################







################################################
################################################################
############################# ARIMA  and Prediction####################################
####################################################################
######################################################################
#from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from pandas import DataFrame

#from statsmodels.tsa.arima_model.ARMAResults import forcast
series = pd.read_csv('1.csv', header=0, index_col=0, parse_dates=True, squeeze=True)

# Changing the name of Wind Speed Column
series = series.rename(columns = {"100m wind speed (m/s)":"WindSpeed"})



# Taking only WindSpeed Column into Consideration
ARIMA_Data = series["WindSpeed"]
X = ARIMA_Data.values 
size_Train = int(len(X)*0.8)
#train,test = X[0:size_Train],X[size_Train:len(X)]
train,test = X[0:30000],X[30000:30010]  ## use if u want walk forward


history = [x for x in train]

#history = history.astype("float32")  #####
predictions = list()

#The result of the forecast() function is an array containing the 
#forecast value, the standard error of the forecast, 
#and the confidence interval information.


#In this part of the code, we are defining our model, fitting it and forcasting.


model = ARIMA(history,order=(2,1,7))
model_fit = model.fit()    
residuals = DataFrame(model_fit.resid)
residuals.plot()
plt.show() 
   

##########################
# find predicted values: for trained data, to check for overfitting
predictions = model_fit.predict(start=0, end=len(train), dynamic=False)
# plot results
plt.figure()
plt.plot(train)
plt.plot(predictions, color='red')
plt.show()

# We need to check if it is overfitting after seeing the result of our test data.
# Bcz the graph shows that the results are so close.




   ################################################33
# density plot of residuals
residuals.plot(kind='kde')
plt.show()
# summary stats of residuals
print(residuals.describe())
# check the RMSE in order to evaluate our forcasting
"""

"""

###
#for train-test
###
model = ARIMA(history,order=(2,1,7))
model_fit = model.fit()
predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)
rmse = math.sqrt(mean_squared_error(test,predictions))
print('Test RMSE: %.3f' % rmse)
# plot forecasts against actual outcomes
plt.plot(test)
plt.plot(predictions, color='red')
plt.show()






## our model is overfitting the training data, so it may be considered to be a bad idea to do
## forcasting for a long term in our case of time series data in wind energy sector.

    

