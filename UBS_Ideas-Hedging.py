#!/usr/bin/env python
# coding: utf-8

# In[16]:


import pandas as pd
import requests
import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller
from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.metrics import mean_squared_error
pd.options.display.max_columns = None
pd.options.display.max_rows = None

import warnings
warnings.filterwarnings("ignore")


# In[17]:


# Data Science Accelerator Credentials 
RESOURCE_ENDPOINT = "https://dsa-stg-edp-api.fr-nonprod.aws.thomsonreuters.com/data/historical-pricing/beta1/views/summaries/" 
access_token = 'tUn7ZD2ZsW9jKiFpoS56ua2g3hnLaUEE8A1nE6cQ'


# In[18]:


ric = 'JPY=' # put the RIC of the asset you want to retrieve data

requestData = {
    'interval': 'P1D',
    'start': '2016-11-01',
    'end': '2019-06-30',
    #"fields": 'TRDPRC_1' # Uncomment this line if you wish to specify which fields to be returned, e.g. TRDPRC_1 is an available field for AAPL.O
};


# In[19]:


def get_data_request(url, requestData):
    """
    HTTP GET request to Refinitiv API
    
    There is more information in the returned dict (i.e. json) object from the API, we store the data in a DataFrame.
    
    :param url: str, the url of the API endpoint
    :param requestData: dict, contains user-defined variables
    :return: DataFrame, containing the historical pricing data. 
        Returned field list order and content can vary depending on the exchange / instrument.
        Therefore returned data must be treated using appropriate parsing techniques.
    """
    dResp = requests.get(url, headers = {'X-api-key': access_token}, params = requestData);       

    if dResp.status_code != 200:
        raise ValueError("Unable to get data. Code %s, Message: %s" % (dResp.status_code, dResp.text));
    else:
        print("Data access successful")
        jResp = json.loads(dResp.text);
        data = jResp[0]['data']
        headers = jResp[0]['headers']  
        names = [headers[x]['name'] for x in range(len(headers))]
        df = pd.DataFrame(data, columns=names )
        return df
    
resource_endpoint_ric = RESOURCE_ENDPOINT + ric  
df = get_data_request(resource_endpoint_ric, requestData)


# In[20]:


print(df.shape)
df.head(10)


# In[21]:


df['DATE'] = pd.to_datetime(df['DATE']).astype('O')

df_train = df[df['DATE']<=pd.to_datetime('2019-06-30')]


# In[22]:


# Change ME!
target = df.BID
    
decomposition = seasonal_decompose(target, freq=24)
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

f = plt.figure(figsize=(16,12))
plt.subplot(411)
plt.plot(target, label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal, label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='best')
plt.tight_layout()


# In[23]:


# Looking for about 99% Confidence
# Check if there is an increasing or decreasing trend in the data

adf_res = adfuller(df['BID'], autolag='AIC')
print('ADF Statistic for %s: %f' % ("CLOSE BID", adf_res[0]))
print('p-value: %f' % adf_res[1])
print('Critical Values:')
for key, value in adf_res[4].items():
    print('\t%s: %.3f' % (key, value))


# In[24]:


data = df_train[['DATE', 'BID']]
data.set_index('DATE', inplace=True)
data = data.resample('D',label='right').ffill()

es_model = ExponentialSmoothing(data, trend='add' , damped=True, seasonal='add', seasonal_periods=325)
es_results = es_model.fit()

# Forecasting until 11-30, if the number is changed here, date_range has to be changed as well.
pred= es_results.forecast(153)
xhat = pred.get_values()

# 2 std
z = 1.96
sse = es_results.sse

predint_xminus = xhat - z * np.sqrt(sse/len(data))
predint_xplus  = xhat + z * np.sqrt(sse/len(data))
date_range = pd.date_range('2019-07-01', '2019-11-30')

# Confidence Interval
upper_pi_data = pd.DataFrame( 
    data  = predint_xplus, 
    index = date_range)

lower_pi_data = pd.DataFrame( 
    data  = predint_xminus, 
    index = date_range)


# In[25]:


plt.figure(figsize=(20,9))
plt.plot(data[data.index>'2016-11-01'], label='Original', color='blue')
plt.plot(pred, label='Predicted', color='red')
plt.fill_between(date_range, predint_xminus, predint_xplus, facecolor='green', alpha=0.5)
plt.legend(loc='best')
plt.title("Overall Trend Chart")
plt.show()


# ## Buy/Sell Strategy

# In[26]:


plt.figure(figsize=(20,9))
plt.plot(pred, label='Predicted', color='red')

plt.fill_between(date_range-2, predint_xminus, predint_xplus, facecolor='green', alpha=0.5)
plt.legend(loc='best')
plt.title("Forecast on JPY/USD")

pred_data = pd.DataFrame(pred.index, columns = ['date'], index=pred.index)
pred_data['value'] = pred.values

loess = lowess(pred_data['value'], pred_data.index, frac=0.15)
pred_data['loess'] = [row[1] for row in loess]
plt.plot(pred_data['loess'], label='LOESS', color='yellow')

pred_data['grad'] = np.gradient(pred_data['loess'], 8)
pred_data['turn'] = 0

f = pred_data.index.freq

# Store Max/Min
for i in pred_data.index:
    if i < pd.to_datetime('2019-11-28'):
        if pred_data['grad'].loc[i] >= 0 and pred_data['grad'].loc[i+ f] <= 0:
            pred_data['turn'].loc[i] = 1
        elif pred_data['grad'].loc[i] <= 0 and pred_data['grad'].loc[i+ f] >= 0:
            pred_data['turn'].loc[i] = 2

for i in pred_data.index:
    if (pred_data['turn'].loc[i] == 1):
        plt.scatter(i, pred_data['loess'].loc[i], color='cyan', s=130)
    elif (pred_data['turn'].loc[i] == 2):
        plt.scatter(i, pred_data['loess'].loc[i], color='magenta', s=130)
plt.show()


# In[27]:


mu = df_train['BID'].mean()
sig = df_train['BID'].std()

pred_data['z-score'] = (pred_data['value'] - mu) / sig


# In[28]:


pred_data.head(15)


# In[29]:


turns_data = pred_data.iloc[np.where((pred_data['turn'] == 1)|(pred_data['turn'] == 2))[0]]

tris = turns_data[['date', 'value', 'turn']].values

outputs = []

for i in range(tris.shape[0] - 1):

     if tris[i][2] == 1 and tris[i+1][2] == 2:
        dt = str(abs(tris[i][0] - tris[i+1][0]))
        dt = dt.split(" ")[0]
        t = str(tris[i][0]).split()[0]
        earnings = abs(tris[i][1] - tris[i+1][1])
        
        outputs.append([t, "FX_OPTION", tris[i][1], tris[i+1][1], earnings, dt])
     elif tris[i][2] == 2 and tris[i+1][2] == 1:
        dt = str(abs(tris[i][0] - tris[i+1][0]))
        dt = dt.split(" ")[0]
        t = str(tris[i][0]).split()[0]
        earnings = abs(tris[i][1] - tris[i+1][1])
        outputs.append([t, "FX_FOWARD", tris[i][1], tris[i+1][1], earnings, dt])
        
for r in outputs:
    if int(r[5]) < 7:
        outputs.remove(r)
    elif int(r[5]) > 90:
        r[5] = 90

output_df = pd.DataFrame(outputs, columns = ['DATE_TRADE', 'TYPE_TRADE', 'ESTM_TURNPT', 'ESTM_ENDPT', 'ESTM_EARNING', 'DURATION(DAYS)'])

predictions_df = pred_data[['value']]


# Output 
print("Storing Outputs...")
output_df.to_excel("output.xlsx")
predictions_df.to_excel("predictions.xlsx")
print("Success!")

# Estimated Earnings
units = 1000

forward = output_df.iloc[np.where(output_df['TYPE_TRADE'] == "FX_FOWARD")[0]]
option = output_df.iloc[np.where(output_df['TYPE_TRADE'] != "FX_FOWARD")[0]]

forward.head()

earnings = (np.sum(forward['ESTM_ENDPT'] - forward['ESTM_TURNPT']) + np.sum(option['ESTM_TURNPT'] - option['ESTM_ENDPT']))* units

print("Total Earnings: " + str(earnings) + " USD")


# In[ ]:




