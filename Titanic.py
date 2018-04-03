
# coding: utf-8

# In[1]:


import pandas as pd
from pandas import Series, DataFrame


# In[2]:


#lets read the data
titanic_df = pd.read_csv('C:/Users/marga/Noteboooks/.ipynb_checkpoints/train.csv', low_memory = False)


# In[3]:


#Lets see the preview of the data
titanic_df.head()


# In[4]:


#Another way to quickly have a view of the data is 
titanic_df.info()


# In[5]:


#All good data analysis projects start off with trying to answer questions
#1) Who were the passengers on Titanic?
#2) What deck were the passengers on and how does that relate to their class?
#3) Where did the passengers come from?
#4) Who was alone and who was with family?

#Dig depeer and who was with family?
#5) What factors led someone survive the sinking?


# In[9]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')
#so that we can see everything on the notebook


# In[16]:


#Who are the passengers on Titanic?

#sns.factorplot('Sex',data=titanic_df)

titanic_df["Sex"].value_counts().plot.bar()

plt.show()


# In[ ]:


#Thus there are more male than female


# In[26]:



sns.factorplot("Sex",data=titanic_df, kind = "count")


# In[28]:



sns.factorplot("Survived",data=titanic_df, kind = "count")


# In[29]:


sns.factorplot("Sex",data=titanic_df, kind = "count", hue = 'Pclass')


# In[30]:


def male_female_child(passenger):
    age, sex = passenger
    
    if age < 16:
        return 'child'
    else:
        return sex


# In[31]:


titanic_df['person'] = titanic_df[['Age', 'Sex']].apply(male_female_child, axis =1)
#Creating a new column, grabbing age and sex column and applying the function


# In[32]:


titanic_df.head(10)


# In[34]:


sns.factorplot('Pclass', data=titanic_df, kind = 'count', hue = 'person')


# In[37]:


titanic_df['Age'].hist(bins = 70)


# In[38]:


titanic_df['Age'].mean()


# In[42]:


titanic_df['person'].value_counts()


# Part 2
# 

# In[47]:


fig = sns.FacetGrid(titanic_df, hue = 'Sex', aspect = 4)
fig.map(sns.kdeplot, 'Age', shade = True)
oldest = titanic_df['Age'].max()
fig.add_legend()
fig.set(xlim=(0, oldest))


# In[53]:


fig = sns.FacetGrid(titanic_df, hue = 'person', aspect = 4)
fig.map(sns.kdeplot, 'Age', shade = True)
oldest = titanic_df['Age'].max()
fig.add_legend()
fig.set(xlim=(0, oldest))


# In[54]:


titanic_df.head()


# In[55]:


deck = titanic_df['Cabin'].dropna()


# In[58]:


deck.head()


# In[63]:


levels = []
for level in deck:
    levels.append(level[0])
    
#What are we doing here? we are creating an empty list and now for every level in levels we are grabbing the first 
#letter that is C, C, E, G, C etc and appending it to levels

cabin_df = DataFrame(levels)
cabin_df.columns = ['Cabin']
sns.factorplot('Cabin', data=cabin_df, palette = 'winter_d', kind = 'count')


# In[66]:


cabin_df = cabin_df[cabin_df.Cabin!= 'T']
sns.factorplot('Cabin', data = cabin_df, palette = 'summer', kind = 'count')


# In[67]:


titanic_df.head()


# In[68]:


titanic_df['Embarked'].unique()


# In[73]:


sns.factorplot('Embarked', data = titanic_df, kind = "count", hue = 'Pclass', order = ['C', 'Q', 'S'])


# # Who is alone and who is with family?

# In[ ]:


#SibSp = whether they had siblings on board
#Parch = whether they had parents or children on board
#If both are 0 then they are completely alone, no parents, children or siblings
#Lets make a new column called find alone as Alone


# In[74]:


titanic_df['Alone'] = titanic_df.SibSp + titanic_df.Parch


# In[90]:


titanic_df['Alone']


# In[ ]:


titanic_df['Alone'].loc[pd.to_numeric(titanic_df['Alone']) > 0] = 'With Family'


# In[91]:



titanic_df['Alone'].loc[titanic_df['Alone'] == 0] = 'Alone'


# In[92]:


titanic_df['Alone']


# In[93]:


titanic_df.head()


# In[96]:


sns.factorplot('Alone', data=titanic_df, palette = 'Blues', kind = 'count')


# In[98]:


titanic_df['Survivor'] = titanic_df.Survived.map({0:'no', 1:'yes'})


# In[100]:


sns.factorplot('Survivor', data=titanic_df, palette = 'Set1', kind = 'count')


# In[102]:


sns.factorplot('Pclass', 'Survived', hue = 'person', data = titanic_df)


# In[104]:


sns.lmplot('Age', 'Survived', data = titanic_df)


# In[106]:


sns.lmplot('Age', 'Survived',hue = 'Pclass', data = titanic_df, palette = 'winter')


# In[107]:


generations = [10, 20, 40, 60, 80]
sns.lmplot('Age', 'Survived', hue = 'Pclass', data = titanic_df, palette = 'winter', x_bins = generations )


# In[108]:


#How does gender and age relate with suvival?


# In[109]:


sns.lmplot('Age', 'Survived', hue = 'Sex', data=titanic_df, palette = 'winter', x_bins = generations)


# In[ ]:


#if youtre an older female you hace better vhances of survival compared to older male


# 1) Did the deck have an effect on the passengers survival rate? Did this answer match up to your iuntuition?
# 
# 2) Did having a family member increase the odds of surviving the crash?

# In[110]:


titanic_df.head()


# In[111]:


sns.factorplot('Survivor', data=titanic_df, hue = 'Alone', palette = 'Set1', kind = 'count')


# In[ ]:


titanic_df.to_csv('C:/Users/marga/Noteboooks/.ipynb_checkpoints/titanicModified.csv')

