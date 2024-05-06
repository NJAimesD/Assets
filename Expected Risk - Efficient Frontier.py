#!/usr/bin/env python
# coding: utf-8

# In[ ]:


###pip install yfinance
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


#download the list of all tickers in S&P
def list_wikipedia_sp500() -> pd.DataFrame:
    # Ref: https://stackoverflow.com/a/75845569/
    url = 'https://en.m.wikipedia.org/wiki/List_of_S%26P_500_companies'
    return pd.read_html(url, attrs={'id': 'constituents'}, index_col='Symbol')[0]


# In[ ]:


df = list_wikipedia_sp500()
symbolslist = df.index.to_list()
#type(symbolslist)

symbolslist.remove("BRK.B")
symbolslist.remove("BF.B")
#Manually removed the two stocks that were not available for download


# In[ ]:


#download all the quotes for those tickers
SPInfo = yf.download(symbolslist, start = '2021-01-01')


# In[ ]:


#Only mantain adj price
SPPrices = SPInfo.drop(["Open","Low","Close","High", "Volume"], axis = 1)
#remove everything but adj close pirce
#print(SPPrices)
SPPrices.info()


# In[ ]:


#to calculate compound returns per stock
returnslog = np.log(SPPrices)
compoundedreturns = returnslog.diff()
#it shows the difference between two consecutive values
#print(compoundedreturns)


# In[ ]:


#calculate expected returns per stock
cretunrsmean = compoundedreturns.dropna().mean(axis=0)
cretunrsmeandf = cretunrsmean.to_frame()
expectedreturn = (np.exp(cretunrsmeandf))-1
#print(expectedreturn)
#type(expectedreturn)


# In[ ]:


#function to create weight matrix

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


# In[ ]:


#create weight matrix for 10 portfolios
mweights = createweightmatrix(10)
#print(mweights)


# In[ ]:


#choose the 10 stocks with the higher expected returns. 
expectedreturn = expectedreturn.rename({0: 'expected return per stock'}, axis=1)
highexreturn = expectedreturn['expected return per stock'].nlargest(n=10)
highexreturn = pd.DataFrame(highexreturn,columns = ['expected return per stock'])
print(highexreturn)
#type(highexreturn)


# In[87]:


#create the EXPECTED RETURN for the 10 portfolios 

#erp = mweights.multipy(highexreturn['expected return per stock', axis=1])
#erp = mweights.mul(highexreturn,1)
erp = np.sum(highexreturn.values * mweights)
print(erp)


# In[ ]:


#EVARPORT=t(W)%*%COV%*%W
#ERISK=sqrt(diag(EVARPORT))
#ERISK


# In[85]:


#retrieve the compund returns of the stock we choose on highexreturns to calculate risk. 

#highexreturn.index
#def asset_list (n): 
cols = [x[1] for x in highexreturn.index]
#compoundedreturns
#compoundedreturns.columns = [x[1] for x in compoundedreturns.columns]
compoundedreturns = compoundedreturns[cols]
compoundedreturns.head()


# In[86]:


#generate the covariance matrix for the 10 stocks we choose

covreturns = compoundedreturns.cov()
print(covreturns)


# In[ ]:


#generates the expected risk for a single portfolio

#np.array(mweights)*np.array(covreturns)
varianceport = np.sqrt(pd.DataFrame(np.dot(np.dot(np.array(mweights), np.array(covreturns)), np.array(mweights.T))))


# In[ ]:


#generate the expected risk for the 10 portfolios

sigmas = []
for i in range(10):
  sigma = varianceport[i][i]
  sigmas.append(sigma)


# In[ ]:


#Graph the efficient frontier

sigmas(X) erp(Y)

