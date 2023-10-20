#!/usr/bin/env python
# coding: utf-8

# In[31]:


###pip install yfinance
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


#download the list of all tickers in S&P
def list_wikipedia_sp500() -> pd.DataFrame:
    # Ref: https://stackoverflow.com/a/75845569/
    url = 'https://en.m.wikipedia.org/wiki/List_of_S%26P_500_companies'
    return pd.read_html(url, attrs={'id': 'constituents'}, index_col='Symbol')[0]


# In[3]:


df = list_wikipedia_sp500()
symbolslist = df.index.to_list()
#type(symbolslist)

symbolslist.remove("BRK.B")
symbolslist.remove("BF.B")
#Manually removed the two stocks that were not available for download


# In[4]:


SPInfo = yf.download(symbolslist, start = '2021-01-01')
#download all the quotes for those tickers


# In[20]:


#create returns
SPPrices = SPInfo.drop(["Open","Low","Close","High", "Volume"], axis = 1)
#remove everything but adj close pirce
#print(SPPrices)
SPPrices.info()


# In[7]:



#to calculate compound returns per stock
returnslog = np.log(SPPrices)
#calculates the natural log
#print(returnslog)
compoundedreturns = returnslog.diff()
#it shows the difference between two consecutive values
print(compoundedreturns)


# In[12]:


#calculate expected returns
cretunrsmean = compoundedreturns.dropna().mean(axis=0)
cretunrsmeandf = cretunrsmean.to_frame()
expectedreturn = (np.exp(cretunrsmeandf))-1
print(expectedreturn)
type(expectedreturn)


# In[9]:


#calcualate risk(std dev) per stock
riskstock = compoundedreturns.std()
print(riskstock)


# In[15]:


def get_weights(n):
  """
    returns a vector of size n, with weights, the sum should be 1.
  """

 

  search_space = np.linspace(0, 1, 1_000_000)
  cumulative_weights = 0
  vector_weight = []

  for i in range(n - 1):
    weight = np.random.choice(list(search_space)) ### uniform distribution.
    vector_weight.append(weight)
    cumulative_weights = cumulative_weights + weight
    search_space = np.linspace(0, 1 - cumulative_weights, 1_000_000)

  last_weight = 1 - cumulative_weights
  vector_weight.append(last_weight)
  return vector_weight

 

def createweightmatrix(n):

    lista1 = []

    for i in range(n):
        lista0 = get_weights(n)
        lista1.append(lista0)
    dataframe1 = pd.DataFrame(lista1, columns=['weigth1','weigth2','weigth3','weigth4','weigth5','weigth6','weigth7','weigth8','weigth9','weigth10'] )

 

    return dataframe1

 

mweights = createweightmatrix(10)
#print(mweights)

 


# In[17]:


expectedreturn = expectedreturn.rename({0: 'expected return per stock'}, axis=1)
highexreturn = expectedreturn['expected return per stock'].nlargest(n=10)
highexreturn = pd.DataFrame(highexreturn,columns = ['expected return per stock'])
print(highexreturn)
type(highexreturn)


# In[30]:


exriskhighreturn  = highexreturn.std()
print(exriskhighreturn)


# In[19]:


cov = highexreturn.cov()
print(cov)


# In[28]:


#highexreturn * mweights

#erp = mweights.multipy(highexreturn['expected return per stock', axis=1])
#erp = mweights.mul(highexreturn,1)
erp = np.sum(highexreturn.values * mweights)
print(erp)


# In[ ]:




