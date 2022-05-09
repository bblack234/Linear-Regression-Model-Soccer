#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#required packages with versions
"""
!pip install numpy==1.21.2
!pip install seaborn==0.11.1
!pip install matplotlib==3.4.3
!pip install statsmodels==0.12.2
!pip install pandas==1.2.4
!pip install scipy==1.6.3
!pip install scikit_learn==0.24.2
"""


# In[3]:


#Package imports needed
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import seaborn as sns
from scipy import stats
import scipy


#Load the data as a data frame
df=pd.read_csv('EPL_Soccer_MLR_LR.csv')

# Get basic description of the data, looking the spread of the different variables,
# along with  abrupt changes between the minimum, 25th, 50th, 75th, and max for the different variables
df.describe()


# In[4]:


#Get info, look for missing values, get a sense of what format each variable is in


df.info()
#We are attempting to predict score
#Look at correlations between variables to identify best predictor for response (score)
df.corr()

#Can see the strongest predictor of score is cost, with a 96% correlation


# In[272]:


#Let's plot cost vs. score
plt.scatter(df['Cost'], df['Score']);

#Strong linear association between cost and score, maybe some concern with model
# after a cost of 125 or so!
