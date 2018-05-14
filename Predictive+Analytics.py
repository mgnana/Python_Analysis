
# coding: utf-8

# # Identifying factors that affect sales  

# # Problem Statement : The franchisee of Store B, Mr. Choe, is interested in identifying the factors that affect sales in his store and the extent to which these factors affect sales.

# Seoul Retail Case
# Case Study
# Store B is located in an exclusive shopping district in Seoul, South Korea. Mr. Choe would like to know about the factors that affect sales in each store and the importance of each factor. Mr. Choe plans to use these identified factors for sales and operations planning in his store. 
# Process 
# To study the impact of the different variable for store B, we split the dataset based on each store and performed predictive modeling separately.  
# The independent variables are - Store ID, Store Name, Number of Customers, Number of Items Sold, Discount, Average Sales per Customer, Average Sales per Item, Date, Day of the week, Distances from Metro Stations X and Y, Distance from the nearest main thoroughfare.
# The dependent variable is Total Sales.
# Linear Regression was used to study the impact of the independent variables on the dependent variable separately for every store.
# 
# 
# Data Preparation
# Feature Engineering
# Categorical Variables
#  Dummy coding is a commonly used method for converting a categorical input variable into a continuous variable. ‘Dummy,’ as the name suggests is a duplicate variable which represents one level of a categorical variable. Presence of a level is represented by 1 and absence is represented by 0. For every level present, one dummy variable will be created.
# We used dummy coding for categorical variables like Outlook, Months and Weekdays to study the granular impact of every level. We used the python library pandas.get_dummies to achieve this.
# Variable Transformation
# 1) IMP_Japanese_Tourists is skewed to the right, so we applied log transformation
# 2) Discount is skewed to the right, so we applied log transformation.
# 3) YenWonRatio  is skewed to the l, so we applied the square transformation
# 
# Variable Imputation
# The Japanese tourist variable has many missing values. We have imputed them using Means through the Tree method. 
# Rejected variables
# No_of_Customers, No_of_Items, Avg_Sales_per_Customer, Avg_Sales_per_Item, Code were rejected since they do not convey any impactful insight as a predictor. 
# Distance_from_Main_Street_Meter, Distance_from_Station_X_Meter,Distance_from_Station_Y_Meter were rejected for Store B. 
# 

# In[3]:


import pandas as pd 
from pandas import DataFrame, Series
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[4]:


#Loading Seoul data set 


# In[5]:


korea_df = pd.read_csv('C:/Users/marga/Noteboooks/.ipynb_checkpoints/korea_data.csv', low_memory = False)


# In[6]:


korea_df.head()


# What are the unique store names?

# In[7]:


korea_df['Store Name'].unique()


# We have 5 stores, lets filter for only Store B

# In[8]:


korea_df.columns


# In[9]:


is_storeB = korea_df['Store Name'] == 'Store B'


# In[10]:


is_storeB.value_counts()


# In[11]:


korea_storeB = korea_df[is_storeB]
print(korea_storeB.shape)


# We have successfully filtered pandas dataframe based on values of a colum. Here all the values of the variable Store Name will be Store B 

# In[12]:


korea_storeB.head()


# In[13]:


#Checking to see the unique values of store name contains only store B


# In[14]:


korea_storeB['Store Name'].unique()


# Checking for the unique values of 'Outlook' column

# In[15]:


korea_storeB['Outlook'].unique()


# Here we can see that the values are unstandardised. Lets make cloudy = Cloudy and rainy = Rainy to ensure the data quality

# In[16]:


korea_storeB['Outlook'] = korea_storeB['Outlook'].replace(['rainy'], 'Rainy')


# In[17]:


korea_storeB['Outlook'].unique()
#Here we can see that 'rainy' has successfully been replaced by 'Rainy'


# In[18]:


korea_storeB['Outlook'] = korea_storeB['Outlook'].replace(['cloudy'], 'Cloudy')


# In[19]:


korea_storeB['Outlook'].unique()
#Here we can see that 'cloudy' has successfully been replaced by 'Cloudy'


# In[20]:


sns.factorplot("Outlook",data=korea_storeB, kind = "count")


# Here we can see that the count of Sunny days is more than that of cloudy, rainy or snowy days

# Lets now have a look at the distribution of Japanese tourists in the dataset. Since its a continuous dataset, lets use a histogram to represent this.

# In[21]:


histo = korea_storeB['Japanese Tourists']


# In[22]:


korea_storeB.head()


# In[23]:


korea_storeB.dtypes


# Here we can see that Japanese Tourists are of the type 'object', and lets convert this into the type 'float'

# In[24]:


seri_JapanTourists= pd.Series(korea_storeB['Japanese Tourists'])


# In[25]:


korea_storeB['Japanese Tourists'] = pd.to_numeric(seri_JapanTourists, errors = 'coerce')


# In[26]:


korea_storeB.dtypes


# Similarly, converting YenWonRatio to 'float' type since it is of the type 'object'

# In[27]:


seri_YenWonRatio= pd.Series(korea_storeB['YenWonRatio'])


# In[28]:


korea_storeB['YenWonRatio'] = pd.to_numeric(seri_YenWonRatio, errors = 'coerce')


# In[30]:


korea_storeB['YenWonRatio'].dtypes


# Thus we have successfully converted 'YenWonRatio' and 'Japanese Tourists' into the type 'float'

# Lets now check how many null values are there in Japanese tourists and their max and min

# In[31]:


korea_storeB['Japanese Tourists'].isnull().sum()


# In[135]:


korea_storeB['Japanese Tourists'].max()


# In[136]:


korea_storeB['Japanese Tourists'].min()


# Lets handle the null values by replacing them with the mean of the values of 'Japanese Tourists'. While taking the mean it is important to note that it excludes NA/null values when computing the results

# In[137]:


korea_storeB['Japanese Tourists'].mean()


# Lets replace the null values with the mean = 3160.65705128201513 using the fillna() method

# In[32]:


korea_storeB['Japanese Tourists'] = korea_storeB['Japanese Tourists'].fillna(korea_storeB['Japanese Tourists'].mean())


# In[33]:


korea_storeB['Japanese Tourists'].isnull().sum()


# Thus we can see above that the count of null values is 0 and that we have replaced all the above null values with the mean of the value of Japanese Tourists

# In[34]:


korea_storeB['YenWonRatio'].isnull().sum()


# In[35]:


plt.hist(korea_storeB['YenWonRatio'])
plt.show()


# Lets use square transforamtion to reduce left skewness of the 'YenWonRatio'

# In[36]:


korea_storeB['YenWonRatio'].apply(np.log).hist()

plt.show()


# In[144]:


korea_storeB['YenWonRatio'] = np.sqrt(korea_storeB['YenWonRatio'])


# In[37]:


plt.hist(korea_storeB['YenWonRatio'])
plt.show()


# In[38]:


plt.hist(korea_storeB['Japanese Tourists'])
plt.show()


# 'Japanese Tourists' is right skewed and right skewness can be handled by a logarithmic transformation

# In[39]:


korea_storeB['Japanese Tourists'] = np.log(korea_storeB['Japanese Tourists'])


# In[40]:


plt.hist(korea_storeB['Japanese Tourists'])
plt.show()


# In[109]:


korea_storeB['Japanese Tourists'].min()


# In[110]:


korea_storeB['Japanese Tourists'].max()


# In[41]:


korea_storeB.head()


# In[42]:


dummy = pd.get_dummies(korea_storeB['Outlook'])
dummy.head()


# In[43]:


korea_storeB = pd.concat([korea_storeB, dummy], axis = 1)
korea_storeB.head()


# In[44]:


korea_storeB.columns


# In[45]:


dummy_week = pd.get_dummies(korea_storeB['Weekday'])
dummy_week.head()


# In[46]:


korea_storeB = pd.concat([korea_storeB, dummy_week], axis = 1)
korea_storeB.head()


# In[47]:


dummy_month = pd.get_dummies(korea_storeB['Month'])
dummy_month.head()


# In[48]:


dummy_month = dummy_month.rename(columns={1: 'Jan', 2: 'Feb', 3: 'March', 4: 'April', 5: 'May',
                            6: 'June', 7: 'July', 8: 'Aug', 9:'Sept', 10 :'Oct',
                            11: 'Nov', 12: 'Dec'})


# In[49]:


dummy_month.head()


# In[50]:


korea_storeB['Month'].dtypes


# In[51]:


korea_storeB = pd.concat([korea_storeB, dummy_month], axis = 1)
korea_storeB.tail()


# In[52]:


korea_storeB_droppedColumns = korea_storeB.drop(['Month', 'Weekday', 'Outlook'], axis = 1)


# In[53]:


korea_storeB_droppedColumns.columns


# # Feature Importance

# In[54]:


#feature names as a list

col = korea_storeB_droppedColumns.columns
print(col)


# Dropping the features that are not very important. Howver, here I am dropping other columns which are quite intutive that may affect the sales directly like '#of Customers' and '# of items'. These features are quite obvious that increase the total sales. Hence dropping them. This gives us a better idea on what other features tend to affect sales

# In[55]:


#y includes our labels and x includes our features
list = ['Code', '# of Customers','# of Items', 'Total Sales','Store Name', 'Avg Sales per Customer', 'Avg Sales per Item', 'Date',
        'Year', 'Distance from Station X(Meter)', 'Distance from Station X(Feet)', 'Distance from Station Y(Meter)',
        'Distance from Station Y(Feet)', 'Distance from Main Street(Meter)', 'Distance from Main Street(Feet)']
x = korea_storeB_droppedColumns.drop(list,axis = 1 )

y = korea_storeB_droppedColumns['Total Sales']
x.head()


# In[56]:


plt.hist(y)
plt.show()


# Here we can see the distribution of our total sales

# What if we want to observe all correlation between features? Yes, you are right. The answer is heatmap that is old but powerful plot method.

# In[57]:


#correlation map
f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(x.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)


# From the heat map above we can see that '# of customers' and '# of items' are correlated with 'Total Sales' with a correlation coefficient equivalent to 0.8

# Therefore lets eliminate '# of customers' and '# of items' and lets use random forest and find accuracy according to chosen features.

# In[58]:


drop_list1 = ['# of customers','# of items']
x_1 = x.drop(drop_list1,axis = 1 )        
x_1.head()


# Lets find the next highest predictors using random forest

# In[59]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score,confusion_matrix
from sklearn.metrics import accuracy_score

# split data train 70 % and test 30 %
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)




# In[60]:


from sklearn.linear_model import LinearRegression

regression_model = LinearRegression()
regression_model.fit(x_train, y_train)


# In[61]:


for idx, col_name in enumerate(x_train.columns):
    print("The coefficient for {} is {}".format(col_name, regression_model.coef_[idx]))


# In[62]:


intercept = regression_model.intercept_

print("The intercept for our model is {}".format(intercept))


# # Scoring Model

# A common method of measuring the accuracy of regression models is to use the R squared statistic

#  R2 can be determined using our test set and the model’s score method.

# In[63]:


regression_model.score(x_test, y_test)


# So in our model, 47.61% of the variability in Y can be explained using X

# We can also get the mean squared error using scikit-learn’s mean_squared_error method and comparing the prediction for the test data set (data not used for training) with the ground truth for the data test set:

# In[64]:


from sklearn.metrics import mean_squared_error

y_predict = regression_model.predict(x_test)

regression_model_mse = mean_squared_error(y_predict, y_test)

regression_model_mse


# In[65]:


import math

math.sqrt(regression_model_mse)


# So we are an average of 3013458.44 Won away from the ground truth Won total sales when making predictions on our test set.

# In[79]:


from matplotlib import pyplot


# In[66]:


sorted(regression_model.coef_)


# In[107]:



a = []
c = []
for idx, col_name in enumerate(x_train.columns):
    sorted_coefficients = regression_model.coef_[idx]
    a.append(regression_model.coef_[idx])
    c.append(col_name)



# In[113]:


percentile_list = pd.DataFrame(
    {'Columns': c,
     'Values': a,
    })

percentile_list.head()



# In[111]:


test = percentile_list.sort_values('Values', ascending=False)


# In[115]:


test


# # Final Analysis

# From the above table we can see that apart from '# of customers' and '# of tourists', the number of 'Japanese Tourists' seem to be one of the major contributors of Total Sales of Store B and we can see that Tuesday's and the month of July tend to negatively affect sales

# # Thank You!
