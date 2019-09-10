#!/usr/bin/env python
# coding: utf-8

# # Parth Patel :   Assignment 2

# In[616]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import cross_val_score 
from sklearn.metrics import classification_report
#display(data)


# In[617]:


data = pd.read_csv('Desktop/Data/ETFs/eem.us.txt', header = None).dropna()


# In[618]:



    # Set up type variable with input

Instrument = str(input('Enter your investment instrument type(ETFs/Stocks): '))
x = str(input('Enter your investment instrument name(small case):'))
data = pd.read_csv("Desktop/Data/"+Instrument+"/"+x+".us.txt", header = None).dropna()

#display(data)
data.columns = ['Date','Open','High','Low','Close','Volume','OpenInt']
data1 = data.iloc[:,0:6].drop(0)     


# In[628]:



data1['Open'] = pd.to_numeric(data1['Open'][1:])
data1['High'] = pd.to_numeric(data1['High'][1:])
data1['Low'] = pd.to_numeric(data1['Low'][1:])
data1['Close'] = pd.to_numeric(data1['Close'][1:])
data1['Volume'] = pd.to_numeric(data1['Volume'][1:])
data1.insert(6,"Lag Price", data1['Close'].shift(-1))
ZLEMA = (data1['Close'] - data1["Close"].shift(-1)) + data1['Close']
    # data1['ZLEMA'] = data1.apply(lambda row: (row.Close - row.Close.shift(-1)) + row.Close , axis = 1)
data1.insert(7, "ZLEMA", ZLEMA)
data1['ZLEMA'].fillna(0)
data1['Lag Price'].fillna(0)
    


# In[629]:


data2 = data1[1:][0:3200]


# In[630]:


data2.shape


# In[631]:


Y = data2['Open'][0:3199]
X = data2[['High','Low','Close','Volume','Lag Price','ZLEMA']]


# In[520]:


Y.shape


# In[632]:


index = X['ZLEMA'].index[X['ZLEMA'].apply(np.isnan)]
df_index = X.index.values.tolist()
[df_index.index(i) for i in index]


# In[633]:


X.isnull().sum()


# In[634]:


X1 = X.dropna()
X1.isnull().sum()


# In[595]:


X1.shape


# In[635]:


type(X1['Lag Price'][3])


# In[636]:


X2 = X1


# Linear model to predict daily price using Scikit-Learn model

# In[637]:


X_train, X_test, y_train, y_test = train_test_split(X1, Y, test_size=0.2, random_state=0)


# In[638]:


reg = LinearRegression().fit(X_train, y_train)
y_pred = reg.predict(X_test)


# In[639]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[640]:


from sklearn.metrics import accuracy_score

print('Explained Variance',metrics.explained_variance_score(y_test, y_pred))
print('R2 value', metrics.r2_score(y_test,  y_pred))


# In[641]:


X1


# In[644]:


for j, i in enumerate(tuple(X1.columns)):
    print("The coefficient for {} is {}". format(i, reg.coef_[j]))
    print("The intercept for our model is {}".format(reg.intercept_))    


# In[604]:


X1.shape


# In[645]:


print("Linear regression: Test score:", reg.score(X1, Y))


# In[646]:


X1.shape


# In[667]:


# data1 = data1[:4].tail(25)
# data1.plot(kind='bar',figsize=(16,10))
# plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
# plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
# plt.show()

#plt.scatter(y_test, y_pred)
    #As volume is much higher than any other features thus others become negligible, though we can check it using 
    # matplotlib notebook


# In[668]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.scatter(y_test, y_pred,  color='gray')
plt.plot(y_test, y_pred, color='red', linewidth=0.02)
plt.show()


# Linear regression using StatsModels

# In[651]:


import statsmodels.api as sm
import numpy as np
import pandas as pd


# In[652]:


X1.shape


# In[654]:


X1 = pd.DataFrame(X1)
X1.tail(20)


# In[655]:


# Note the difference in argument order
model = sm.OLS(Y, X1).fit()
#X1 = sm.add_constant(X1)
predictions = model.predict(X1) # make the predictions by the model

# Print out the statistics
model.summary()


# Difference between scikit and statmodels is basically scikit model is scikit model is easy and very fast to use on big data. 
# Whereas statmodels were implemented lately by 2017 to get an deep idea about how our data is related to their independednt variables. Statmodels has variety of options for linear regression models. It also gives skewness, Durbin-Watson and several other statistic values to understand 
# our predicted variable.
# 
# In my model, comparing scipy with statmodel, I have come to the conclusion that the positive co-efficients in statmodels are
# more positive comparatively to scipy and negative are more negative. The reason for occuring this situation is the effect of 
# those variables with positive/negative co-efficient is intensive on our predictive variable. Finally, giving us an accurate result.

# 
# 

# Least Square Method 
# 

# In[656]:


import scipy.optimize as optimization
import numpy as np


# In[657]:


X1.shape


# In[658]:


Y.shape


# In[454]:


#params = len(X1.columns)


# In[660]:


def func(params, xdata, ydata):
    return (Y - np.dot(X1, params))


# In[661]:


print(optimization.leastsq(func, np.zeros(6), args=(Y, X1)))


# In[663]:


plt.plot(Y, X1['Lag Price'])
     #predicted dependent feature vs independent values
   


# In[ ]:





# In[664]:


plt.plot(Y, X1['ZLEMA'])  #predicted dependent feature vs independent values


# In[665]:


plt.plot(Y, X1['Close']) #predicted dependent feature vs independent values


# In[666]:


plt.plot(Y, X1['High'])     #predicted dependent feature vs independent values


# In[ ]:





# In[ ]:




