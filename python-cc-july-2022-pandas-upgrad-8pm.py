#!/usr/bin/env python
# coding: utf-8

# In[82]:


print('I like Pandas')


# In[7]:


# ModuleNotFoundError: No module named 'numpy'
get_ipython().system('pip install numpy')

# ModuleNotFoundError: No module named 'pandas'
get_ipython().system('pip install pandas')


# In[6]:


lst = [10,20,30]
print(lst, type(lst))

import numpy as np

arr = np.array(lst)
print(arr, type(arr))

import pandas as pd


# In[8]:


lst = [10,20,30]
print(lst, type(lst))

arr = np.array(lst)
print(arr, type(arr))

# Series
srs = pd.Series(data=lst)
print(srs)
print(type(srs))


# In[9]:


lst = [10,20,30.0]
print(lst, type(lst))

arr = np.array(lst)
print(arr, type(arr))

# Series
srs = pd.Series(data=lst)
print(srs)
print(type(srs))


# In[10]:


lst = [10,'20',30.0]
print(lst, type(lst))

arr = np.array(lst)
print(arr, type(arr))

# Series
srs = pd.Series(data=lst)
print(srs)
print(type(srs))


# In[11]:


'a b c'.split()


# In[15]:


lst = [10,20,30]
print(lst, type(lst))

arr = np.array(lst)
print(arr, type(arr))

# Series
srs = pd.Series(data=lst, index=['a', 'b', 'c'])
print(srs)
print(type(srs))
print(srs.index)


# In[17]:


dct = {'a':10, 'b':20,'c':30}
srs = pd.Series(data=dct)
print(srs)
print(type(srs))
print(srs.index)
print(srs.dtype)


# In[22]:


srs = pd.Series(data=[10,20,30], index=['a', 'b', 'c'])
print(srs)
print(srs.index)
# getting values
print(srs['b'])
print(srs.iloc[1])
print(srs.iloc[-2])


# In[28]:


srs = pd.Series(data=[10,20,30], index=[1,2,3])
print(srs)
print(srs.index)
# getting values
print(srs[1])
print(srs.iloc[-2])
print(srs.values)


# In[31]:


# maths
srs = pd.Series(data=[10,20,30], index=['a', 'b', 'c'])
print(srs)

print('>> +')
print(srs+srs)
print(srs+2)
print('>> -')
print(srs-srs)
print(srs-2)
print('>> *')
print(srs*srs)
print(srs*2)
print('>> /')
print(srs/srs)
print(srs/2)


# In[34]:


# maths
srs1 = pd.Series(data=[10,20,30], index=['a', 'b', 'c'])
print(srs1)
srs2 = pd.Series(data=[10,20,30], index=['b', 'c', 'd'])
print(srs2)

print('>> +')
print(srs1+srs2)
print('>> /')
print(srs1/srs2)
print('>> *')
print(srs1*srs2)
print('>> -')
print(srs1-srs2)


# In[37]:


arr = np.random.rand(5,4)
print(arr)


# In[47]:


# DataFrame
df = pd.DataFrame(data = arr, index='a b c d e'.split(), columns='C1 C2 C3 C4'.split())
print(type(df))
print(df.index)
print(df.columns)
print(df.dtypes)
df


# In[43]:


# DataFrame
df = pd.DataFrame(data = arr, index='a b c d e'.split())
print(type(df))
print(df.index)
df


# In[40]:


# DataFrame
df = pd.DataFrame(data = arr)
print(type(df))
# print(df)
df


# In[51]:


# DataFrame
df = pd.DataFrame(data = arr, index='a b c d e'.split(), columns='C1 C2 C3 C4'.split())
# col
col = df['C3']
print(col)
print(type(col))
col = df.C3
print(col)
print(type(col))
df


# In[53]:


# DataFrame
df = pd.DataFrame(data = arr, index='a b c d e'.split(), columns='C1 C2 C3 C4'.split())
df[['C3','C1']]


# In[58]:


df = pd.DataFrame(data = arr, index='a b c d e'.split(), columns='C1 C2 C3 C4'.split())
print(df.C3-df.C1)
df["new col"] = df.C3-df.C1
df


# In[62]:


ddf = df.drop('new col', axis=1)
ddf


# In[63]:


df.drop('new col', axis=1, inplace =True)
df


# In[64]:


df.drop('b', axis=0, inplace =True)
df


# In[65]:


df = pd.DataFrame(data = arr, index='a b c d e'.split(), columns='C1 C2 C3 C4'.split())
df


# In[76]:


# single value eg 0.530434
## col > row
print(df['C3']['b'])
print(df.C3['b'])
print(df['C3'].iloc[1])
print(df.C3.iloc[1])
## row > col
print(df.loc['b']['C3'])
print(df.iloc[1]['C3'])
print(df.loc['b'].iloc[2])
print(df.iloc[1].iloc[2])
print(df.loc['b','C3'])
print(df.iloc[1,2])


# In[70]:


# row
print(df.loc['b'])
print(df.iloc[1])


# In[67]:


# col
print(df['C3'])
print(df.C3)


# In[77]:


df = pd.DataFrame(data = arr, index='a b c d e'.split(), columns='C1 C2 C3 C4'.split())
df


# In[78]:


df.set_index('C3')


# In[79]:


df = pd.DataFrame(data = arr, index='a b c d e'.split(), columns='C1 C2 C3 C4'.split())
df


# In[81]:


df.reset_index(drop=True)


# In[85]:


arr = np.array([1,3,5])
print(arr)
print(arr>4)
print(arr[arr>4])


# In[89]:


df = pd.DataFrame(data = np.random.rand(5,4), index='a b c d e'.split(), columns='C1 C2 C3 C4'.split())
df


# In[93]:


df[df.C4>0.5]


# In[91]:


df[df>0.5]


# In[90]:


# conditionals
df>0.5


# In[100]:


df = pd.DataFrame(data = np.random.rand(5,4), index='a b c d e'.split(), columns='C1 C2 C3 C4'.split())
new_df = df[df>0.5]
new_df


# In[106]:


(0.912091+0.759458)/2


# In[105]:


new_df['C3']=new_df.C3.fillna(value = new_df.C3.mean())
new_df


# In[103]:


new_df.fillna(value = 0)


# In[102]:


new_df.dropna(thresh=2)# at least thresh number of non NAN values should be present


# In[101]:


new_df.dropna()


# In[107]:


data = {
    'Country':['in','in','cn','us','cn'],
    'Flag':['A','B','A','B','A'],
    'Score':[1,2,3,2,1]
}
df = pd.DataFrame(data)
df


# In[116]:


# drop duplicates
df.drop_duplicates(['Country', 'Flag'], )


# In[114]:


# sort the values
df.sort_values(['Score', 'Country'], ascending=[False, True])


# In[110]:


# value count
print(df.Country.value_counts())


# In[109]:


# nunique
print(df.Country.nunique())


# In[108]:


# unique
print(df.Country.unique())


# In[120]:


data = {
    'Country':['in','in','cn','us','cn'],
    'Flag':['A','B','A','B','A'],
    'Score':[1,2,3,2,1]
}
df = pd.DataFrame(data)
df


# In[121]:


def my_sq_func(x):
    return x*x
df['Squared_Score']=df.Score.apply(my_sq_func)
df


# In[124]:


def my_sum_func(a,b):
    return a+b
df['Squared_Sum_Score']=df.apply(lambda df:my_sum_func(a=df.Score,b=df.Squared_Score), axis=1)
df


# In[126]:


df.groupby('Country').sum()


# In[127]:


df.info()


# In[128]:


df.groupby('Country').count()


# In[129]:


df.describe()


# In[ ]:


# SBI
# adhaar, name, bal

# govt 
# adhaar, name, criminal

# >>>
# adhaar, name, bal, criminal


# In[130]:


sbi_df = pd.DataFrame({
'aadhaarNo':["000003", "000005", "000001", "000006", "000002"],
'Name':     ["Tony",   "Thor",   "Natasha", "Fury", "Groot"],
'Balance':  [999999,    100,      200,        300,     0]
})
govt_df = pd.DataFrame({
'identificationNo':["000001", "000002", "000003", "000004", "000005", "000006"],
'Name':            ["Natasha", "Groot",  "Tony",   "Tony",   "Thor",   "Fury"],
'criminal':        [True,       True,      True,    False,   False,    False]
})


# In[131]:


sbi_df


# In[132]:


govt_df


# In[137]:


pd.merge(left = sbi_df, right = govt_df, how='inner', left_on = ['aadhaarNo', 'Name'], right_on = ['identificationNo', 'Name'])


# In[135]:


pd.merge(left = sbi_df, right = govt_df, how='inner', left_on = 'aadhaarNo', right_on = 'identificationNo', suffixes = ("_sbi", "_govt"))


# In[133]:


pd.merge(left = sbi_df, right = govt_df, how='inner', on='Name')

