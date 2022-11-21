#!/usr/bin/env python
# coding: utf-8

# # 911 Calls Capstone Project

# For this capstone project we will be analyzing some 911 call data from [Kaggle](https://www.kaggle.com/mchirico/montcoalert). The data contains the following fields:
# 
# * lat : String variable, Latitude
# * lng: String variable, Longitude
# * desc: String variable, Description of the Emergency Call
# * zip: String variable, Zipcode
# * title: String variable, Title
# * timeStamp: String variable, YYYY-MM-DD HH:MM:SS
# * twp: String variable, Township
# * addr: String variable, Address
# * e: String variable, Dummy variable (always 1)
# 
# Just go along with this notebook and try to complete the instructions or answer the questions in bold using your Python and Data Science skills!

# ## Data and Setup

# ____
# ** Import numpy and pandas **

# In[1]:


import numpy as np
import pandas as pd


# ** Import visualization libraries and set %matplotlib inline. **

# In[2]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# ** Read in the csv file as a dataframe called df **

# In[3]:


df = pd.read_csv('911.csv.zip')


# ** Check the info() of the df **

# In[4]:


df.info()


# ** Check the head of df **

# In[5]:


df.head()


# ## Basic Questions

# ** What are the top 5 zipcodes for 911 calls? **

# In[6]:


df['zip'].value_counts().head(5)


# ** What are the top 5 townships (twp) for 911 calls? **

# In[7]:


df['twp'].value_counts().head(5)


# ** Take a look at the 'title' column, how many unique title codes are there? **

# In[8]:


df['title'].nunique()


# ## Creating new features

# ** In the titles column there are "Reasons/Departments" specified before the title code. These are EMS, Fire, and Traffic. Use .apply() with a custom lambda expression to create a new column called "Reason" that contains this string value.** 
# 
# **For example, if the title column value is EMS: BACK PAINS/INJURY , the Reason column value would be EMS. **

# In[9]:


extract=lambda x: x.split(':')[0]
df=df.assign(Reason=df['title'].apply(extract))


# ** What is the most common Reason for a 911 call based off of this new column? **

# In[10]:


df['Reason'].value_counts()
#EMS


# ** Now use seaborn to create a countplot of 911 calls by Reason. **

# In[11]:


sns.countplot(x='Reason',data=df)
#what palette?


# In[12]:


# Create boxplot, joint and some pairplot


# ___
# ** Now let us begin to focus on time information. What is the data type of the objects in the timeStamp column? **

# In[16]:


type('timeStamp')


# ** You should have seen that these timestamps are still strings. Use [pd.to_datetime](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.to_datetime.html) to convert the column from strings to DateTime objects. **

# In[17]:


df['timeStamp']=pd.to_datetime(df['timeStamp'], format='%Y/%m/%d %H:%M:%S')


# ** You can now grab specific attributes from a Datetime object by calling them. For example:**
# 
#     time = df['timeStamp'].iloc[0]
#     time.hour
# 
# **You can use Jupyter's tab method to explore the various attributes you can call. Now that the timestamp column are actually DateTime objects, use .apply() to create 3 new columns called Hour, Month, and Day of Week. You will create these columns based off of the timeStamp column, reference the solutions if you get stuck on this step.**

# In[18]:


df['Hour']=df['timeStamp'].apply(lambda x: x.hour)
df['Month']=df['timeStamp'].apply(lambda x: x.month)
df['Day of Week']=df['timeStamp'].apply(lambda x: x.day_of_week)


# ** Notice how the Day of Week is an integer 0-6. Use the .map() with this dictionary to map the actual string names to the day of the week: **
# 
#     dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}

# In[19]:


dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
df['Day of Week']=df['Day of Week'].apply(lambda x: dmap[x])


# ** Now use seaborn to create a countplot of the Day of Week column with the hue based off of the Reason column. **

# In[20]:


sns.countplot(x='Day of Week',data=df,hue='Reason')


# **Now do the same for Month:**

# In[21]:


sns.countplot(x='Month',data=df,hue='Reason')


# **Did you notice something strange about the Plot?**
# 
# _____
# 
# ** You should have noticed it was missing some Months, let's see if we can maybe fill in this information by plotting the information in another way, possibly a simple line plot that fills in the missing months, in order to do this, we'll need to do some work with pandas... **

# ** Now create a gropuby object called byMonth, where you group the DataFrame by the month column and use the count() method for aggregation. Use the head() method on this returned DataFrame. **

# In[22]:


byMonth=df.groupby('Month').count()


# In[23]:


byMonth.head()


# ** Now create a simple plot off of the dataframe indicating the count of calls per month. **

# In[24]:


byMonth['twp'].plot()
#Which is the count?


# ** Now see if you can use seaborn's lmplot() to create a linear fit on the number of calls per month. Keep in mind you may need to reset the index to a column. **

# In[25]:


byMonth=byMonth.reset_index()
sns.lmplot(x='Month',y='twp',data=byMonth)


# **Create a new column called 'Date' that contains the date from the timeStamp column. You'll need to use apply along with the .date() method. ** 

# In[26]:


df['Date']=df['timeStamp'].apply(lambda x: x.date())


# ** Now groupby this Date column with the count() aggregate and create a plot of counts of 911 calls.**

# In[27]:


df.groupby('Date').count()['twp'].plot()
plt.tight_layout()


# ** Now recreate this plot but create 3 separate plots with each plot representing a Reason for the 911 call**

# In[28]:


df['Reason'].head()


# In[29]:


df['Reason'].tail()


# In[30]:


df_traffic=df[df['Reason']=='EMS']
df_traffic.groupby('Date').count()['twp'].plot()
plt.title('EMS')


# In[31]:


df_traffic=df[df['Reason']=='Fire']
df_traffic.groupby('Date').count()['twp'].plot()
plt.title('Fire')


# In[32]:


df_traffic=df[df['Reason']=='Traffic']
df_traffic.groupby('Date').count()['twp'].plot()
plt.title('Traffic')


# ____
# ** Now let's move on to creating  heatmaps with seaborn and our data. We'll first need to restructure the dataframe so that the columns become the Hours and the Index becomes the Day of the Week. There are lots of ways to do this, but I would recommend trying to combine groupby with an [unstack](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.unstack.html) method. Reference the solutions if you get stuck on this!**

# In[34]:


df2=df.groupby(['Hour','Day of Week']).count().unstack()['Reason']
df2.head()


# ** Now create a HeatMap using this new DataFrame. **

# In[ ]:


sns.heatmap(df2)


# ** Now create a clustermap using this DataFrame. **

# In[ ]:


sns.clustermap(df2)


# ** Now repeat these same plots and operations, for a DataFrame that shows the Month as the column. **

# In[ ]:


df3=df.groupby(['Month', 'Day of Week']).count().unstack()['Reason']
df3.head()


# In[ ]:


sns.heatmap(df3)


# In[ ]:


sns.clustermap(df3)


# In[ ]:


get_ipython().system('pwd')


# **Continue exploring the Data however you see fit!**
# # Great Job!
