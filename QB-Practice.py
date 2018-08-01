
# coding: utf-8

# In[1]:


import pandas as pd
from pandas import Series, DataFrame
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')

get_ipython().magic('matplotlib inline')


# In[2]:


print ("Let's open and read a short text file")
filename = input("Type the filename: ")

f = open(filename)

print("We opened it. Now we will read it with a print statement. \n")

f1 = f.read()


# In[3]:


f1


# In[4]:


import re


# In[5]:


mail_address = re.findall("\w+@[a-zA-Z]+\.[a-zA-Z.]+", f1)


# In[6]:


mail_address


# In[7]:


email_address = re.findall("\w+@[a-zA-Z]+\.[a-zA-Z]+", f1)


# In[8]:


email_address


# In[10]:


#count occurences
di = {}


# In[11]:


for line in mail_address:
    parts = line.split(",")
    host = parts[0]
    if host in di:
        di[host] += 1
    else:
        di[host] = 1


# In[12]:


di


# # Reading a text file and counting the occurence of each word in the file

# In[13]:


print("Lets read a simple plain text file")

file = input("Type the file name:")

print("Now lets open the file")
fileopen = open(file)

print("After opening, we need to read the file")
fileread = fileopen.read()


# In[14]:


fileread


# In[15]:


import re


# In[16]:


word = re.findall("\w+", fileread)


# In[17]:


word


# In[5]:


import json

data_file = input("Type the file name:")

print("Now lets open the file")
data_file_open = open(data_file)


# In[18]:


data = r'C:/Users/marga/Noteboooks/.ipynb_checkpoints/example_2.json'


# In[6]:


data = json.load(open(data_file))


# In[7]:


quiz_data = data['quiz']


# In[8]:


quiz_data


# In[190]:


# open a file for writing

quizy_data = open('C:/Users/marga/Noteboooks/.ipynb_checkpoints/quizzy_data.csv', 'w')


# In[14]:


data


# In[19]:


import csv
import json
import pandas as pd


# In[20]:


with open(data, 'r') as json_data:
    json_list = json.load(json_data)


# In[27]:


f = csv.writer(open("data.csv", "w"))
f.writerow(["quiz", "maths", "q1", "answer","options","question"])
for items in json_list:
    options = items['options']
    question = items['question']
    f.writerow([quiz,options,question])


# In[192]:


count = 0

for dta in quizy_data:
    if count == 0:
        header = dta.keys()
        csvwriter.writeout(header)
        count +=1
        csvwriter.writerow(dta.values())
quizy_data.close()


# In[187]:


infile = open('C:/Users/marga/Downloads/QB/files/machine_transaction.json', 'r')

outfile = open('C:/Users/marga/Downloads/QB/files/machine_transaction2.csv', 'w')


# In[188]:


import csv

writer = csv.writer(outfile)

for row in json.loads(infile.read()):
    writer.writerow(row)
    


# In[ ]:


count = 0

for dta in quizy_data:
    if count == 0:
        header = dta.keys()
        csvwriter.writeout(header)
        count +=1
        csvwriter.writerow(dta.values())
quizy_data.close()


# In[4]:


data


# In[24]:


data["quiz"]["maths"]["q1"]["answer"]


# In[26]:


equip_df = pd.read_csv('C:/Users/marga/Downloads/QB/files/equipment_lu.csv',  low_memory = False)


# In[27]:


equip_df.head()


# In[28]:


timesheets_df = pd.read_csv('C:/Users/marga/Downloads/QB/files/timesheets.csv',  low_memory = False)


# In[29]:


timesheets_df.head()


# In[30]:


mactran_df = pd.read_csv('C:/Users/marga/Downloads/QB/files/machine_transaction.csv',  low_memory = False)


# In[31]:


mactran_df.head()


# In[150]:


mactran_df['eq_id'].nunique()


# In[151]:


equip_df['eq_id'].nunique()


# In[ ]:


#pd.merge(restaurant_ids_dataframe, restaurant_review_frame, on='business_id', how='outer')


# In[34]:


print(mactran_df.shape)
print(equip_df.shape)


# In[35]:


print(timesheets_df.shape)


# In[36]:


mac_eqip_df = pd.merge(mactran_df, equip_df, on='eq_id', how='outer')


# In[38]:


mac_eqip_df.dropna().head()


# In[44]:


mac_eqip_df.model.dropna().nunique()


# In[45]:


print(timesheets_df.shape)
print(mac_eqip_df.shape)


# In[48]:


combine = [mac_eqip_df]


# In[54]:


for dataset in combine:
    dataset['lname'] = dataset.operator.str.extract('([A-Za-z]+)', expand = False)


# In[55]:


mac_eqip_df.head()


# In[71]:


for dataset in com:
    dataset['name'] = dataset.operator.str.extract('([A-Za-z]+)')


# In[68]:


com = [mac_eqip_df]


# In[72]:


mac_eqip_df.head()


# In[76]:


mac_eqip_df = mac_eqip_df.drop(['lname', 'Lname'], axis = 1)


# In[56]:


mac_eqip_df['lname'] = mac_eqip_df['lname'].str[1:]


# In[77]:


mac_eqip_df.head()


# In[58]:


mac_eqip_time_df = pd.merge(mac_eqip_df, timesheets_df, on='lname', how='left')


# In[60]:


timesheets_df.head()


# In[87]:


#Extracting first character from timesheets fname
timecom = ['timesheets_df']


# In[97]:


timesheets_df['fnamechar'] = timesheets_df['fname'].astype(str).str[0]


# In[98]:


timesheets_df.head()


# In[88]:


for dataset in timecom:
    dataset['fnamechar'] = dataset.fnamechar.str.lower()


# In[89]:


timesheets_df.head()


# In[92]:


timesheets_df['lower_fnamechar'] = map(lambda x: x.lower(), timesheets_df['fnamechar'])


# In[93]:


timesheets_df.head()


# In[95]:


timesheets_df.fnamechar = map(str.lower, timesheets_df.fnamechar)


# In[101]:


timesheets_df['lower_fnamechar'] = timesheets_df['fnamechar'].str.lower()


# In[102]:


timesheets_df.head() 


# In[103]:


timesheets_df = timesheets_df.drop(['fnamechar'], axis = 1)


# In[104]:


timesheets_df['lname_first5'] = timesheets_df['lname'].str[0:5]


# In[105]:


timesheets_df.head()


# In[106]:


timesheets_df['lower_lname_first5'] = timesheets_df['lname_first5'].str.lower()


# In[109]:


timesheets_df.head()


# In[108]:


timesheets_df = timesheets_df.drop(['lname_first5'], axis = 1)


# In[110]:


timesheets_df["name"] = timesheets_df["lower_fnamechar"] + timesheets_df["lower_lname_first5"]


# In[111]:


timesheets_df.head()


# In[112]:


mac_eqip_df.head()


# In[113]:


print(mac_eqip_df.shape)


# In[114]:


print(timesheets_df.shape)


# In[134]:


mac_eqip_timesheets_df_inner = pd.merge(mac_eqip_df, timesheets_df, on=['name', 'transaction_date'], how='inner')


# In[116]:


mac_eqip_timesheets_df_inner.shape


# In[136]:


mac_eqip_timesheets_df_inner.shape


# In[135]:


mac_eqip_timesheets_df_inner.head()


# In[138]:


mac_eqip_timesheets_df_inner = mac_eqip_timesheets_df_inner.drop(['fname','lname', 'clock_off_date_time', 'clock_on_date_time', 'lower_fnamechar','lower_lname_first5'], axis = 1)


# In[140]:


mac_eqip_timesheets_df_inner


# In[123]:


timesheets_df['clock_off_date_time'] = timesheets_df['clock_off_date_time'].str[:-5]


# In[124]:


timesheets_df.head()


# In[125]:


mac_eqip_df.head()


# In[131]:


timesheets_df['transaction_date'] = pd.to_datetime(timesheets_df['clock_off_date_time'])


# In[129]:


mac_eqip_df['transaction_date'] = pd.to_datetime(mac_eqip_df['transaction_date'])


# In[ ]:


new_df = pd.merge(A_df, B_df,  how='left', left_on='[A_c1,c2]', right_on = '[B_c1,c2]')


# In[132]:


timesheets_df.head()


# In[141]:


mac_eqip_timesheets_df_inner.shape


# In[143]:


mac_eqip_time_nonNULL_df = mac_eqip_timesheets_df_inner[pd.notnull(mac_eqip_timesheets_df_inner['model'])]


# In[144]:


mac_eqip_time_nonNULL_df.head()


# In[146]:


mac_eqip_time_nonNULL_df.shape


# In[147]:


mac_eqip_time_nonNULL_df.to_csv('C:/Users/marga/Downloads/QB/files/qbfinalfile.csv')

import csv
import json
import pandas as pd

data=r'/Users/phaneendra/Downloads/machine_transaction.json'

with open(data, 'r') as json_data:
    json_list = json.load(json_data)
    
f = csv.writer(open("data.csv", "w"))
f.writerow(["eq_id", "activity", "unit", "volume","operator","transaction_date"])
for items in json_list:
    eq_id = items['eq_id']
    operator = items['operator']
    transaction_date = items['transaction_date']
    f.writerow([eq_id,items['operation']['activity'],items['operation']['unit'],\
                items['operation']['volume'],operator,transaction_date])
    
df = pd.read_csv('data.csv')
df.head()
# In[1]:


import csv
import json
import pandas as pd


# In[21]:


data_mac = r'C:/Users/marga/Downloads/QB/files/machine_transaction.json'


# In[22]:


data_mactran = json.load(open(data_mac))


# In[23]:


data_mactran


# In[25]:


data_mactran['operation']

