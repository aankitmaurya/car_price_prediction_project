
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
import seaborn as sns
sns.set()



# In[4]:


data = pd.read_csv('//home//ankit//Documents//car_price_prediction.csv')
data


# In[8]:


data.describe(include='all')


# In[9]: Droping a column 


data = data.drop(['Model'],axis = 1)
data


# In[10]:


data.describe(include= 'all')


# 

# data.isnull().sum()

# In[11]: finding all null values in dataframe


data.isnull().sum()


# droping missing values

# In[13]:


data = data.dropna(axis = 0)
data.isnull().sum()


# In[15]:


data.describe(include = 'all')


# exploring the pdfs


# In[16]:


sns.distplot(data['Price'])


# Dealing with outliers

# In[17]:


q = data['Price'].quantile(0.99)
data_1 = data[data['Price']<q]
data_1.describe(include = 'all')


# In[19]:


data_3 = data_1[data_1['EngineV']<6.5]
sns.distplot(data_3['EngineV'])


# In[20]:


q = data['Mileage'].quantile(0.99)
data_2 = data_3[data_3['Mileage']<q]
data_2.describe(include = 'all')


# In[21]:


q = data['Year'].quantile(0.01)
data_4 = data_2[data_2['Year']>q]
data_4.describe(include = 'all')


# In[22]:	


sns.distplot(data_4['Year'])


# resetting the indexes

# In[24]:


data_cleaned = data_4.reset_index(drop = True)
data_cleaned.describe(include = 'all')


# ols



# In[25]:


log_price = np.log(data_cleaned['Price'])
data_cleaned['log_price'] = log_price
data_cleaned


# now the graph is linear and we can drop the original price colummn

# In[26]:


data_cleaned = data_cleaned.drop(['Price'],axis = 1)
data_cleaned


# In[28]:


data_cleaned.columns.values


# we will drop year column becouse it has collinearity 

# In[29]:


data_no_multicollinearity = data_cleaned.drop(['Year'],axis = 1)


# data_no_multicollinearity

# creating dummies   if we have n variables there will be n-1 dummies

# In[31]:


data_with_dummies = pd.get_dummies(data_no_multicollinearity,drop_first = True)
data_with_dummies


# In[32]:


data_with_dummies.columns.values


# In[33]:


cols = ['log_price','Mileage', 'EngineV', 'Brand_BMW', 'Brand_Mercedes',
       'Brand_Mitsubish', 'Brand_Renault', 'Brand_Toyota',
       'Brand_Volkswag', 'Body_hatch', 'Body_other', 'Body_sedan',
       'Body_vagon', 'Body_van', 'Engine Ty_DieselF4334', 'Engine Ty_Gas',
       'Engine Ty_Other', 'Engine Ty_Petrol', 'Registrati_yes']


# In[35]:


data_preprocessed = data_with_dummies[cols]
data_preprocessed.head()


# # creating regression

# In[36]:


targets = data_preprocessed['log_price']
inputs = data_preprocessed.drop(['log_price'],axis = 1)


# # scale the data

# In[39]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(inputs)


# In[40]:


input_scaled = scaler.transform(inputs)


# In[ ]:




# 
# # Train test Split

# In[41]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(input_scaled, targets, test_size=0.2, random_state=365)


# # Creating regression

# In[42]:


reg = LinearRegression()
reg.fit(x_train,y_train)


# In[44]:


y_hat = reg.predict(x_train)


# In[47]:


plt.scatter(y_train, y_hat)
plt.xlabel('Targets (y_train)', size = 18)
plt.ylabel('Predictions (y_hat)',size = 18)
plt.xlim(6,13)
plt.ylim(6,13)
plt.show()


# # residual are difference between target and predictions

# In[48]:


sns.distplot(y_train - y_hat)
plt.title("resedual ", size = 18)


# In[49]:


reg.score(x_train, y_train)


# # finding weights and bias

# In[50]:


reg.intercept_


# In[51]:


reg.coef_


# In[52]:


data_cleaned['Brand'].unique()


# In[53]:


y_hat_test = reg.predict(x_test)


# In[55]:


plt.scatter(y_test, y_hat_test, alpha = 0.2)
plt.xlabel('Targets (y_test)', size = 18)
plt.ylabel('Predictions (y_hat_test)',size = 18)
plt.xlim(6,13)
plt.ylim(6,13)
plt.show()


# In[56]:


df_pf = pd.DataFrame(np.exp(y_hat_test), columns = ['Predictions'])
df_pf.head()


# In[57]:


y_test = y_test.reset_index(drop = True)
y_test.head()


# In[58]:


df_pf['Target'] = np.exp(y_test)
df_pf


# In[60]:


df_pf['Resedual'] = df_pf['Target'] - df_pf['Predictions']
df_pf['Difference%'] = np.absolute(df_pf['Resedual']/df_pf['Target']*100)
df_pf


# In[61]:


df_pf.describe()


# In[ ]:




