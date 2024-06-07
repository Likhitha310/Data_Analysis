#!/usr/bin/env python
# coding: utf-8

# # 1. Loading the Datasets

# In[1]:


import pandas as pd

# Load the datasets
athlete_events_df = pd.read_csv('athlete_events.csv')
noc_regions_df = pd.read_csv('noc_regions.csv')


# # 2. Merging Datasets

# In[2]:


# Merge on 'NOC' column
merged_df = pd.merge(athlete_events_df, noc_regions_df, on='NOC', how='left')


# # 3. Sorting Data

# In[3]:


# Sort by 'Year' and 'Medal'
sorted_df = merged_df.sort_values(by=['Year', 'Medal'], ascending=[True, True])
sorted_df.head()


# # 4. Filtering Data using Regex

# In[4]:


# Filter athletes whose names start with 'A'
filtered_df = merged_df[merged_df['Name'].str.contains('^A', regex=True)]
filtered_df.head()


# # 5. Grouping and Aggregation

# In[5]:


# Group by 'region' and count the number of medals
medals_by_region = merged_df[merged_df['Medal'].notnull()].groupby('region')['Medal'].count().reset_index()
medals_by_region.columns = ['Region', 'Medal Count']

# Sort by 'Medal Count' in descending order
medals_by_region = medals_by_region.sort_values(by='Medal Count', ascending=False)
medals_by_region.head()


# # 6. Handling Missing Values

# In[6]:


# Fill missing values in 'Age' column with the mean age
merged_df['Age'] = merged_df['Age'].fillna(merged_df['Age'].mean())

# Drop rows with any missing values
cleaned_df = merged_df.dropna()
cleaned_df.head()


# # 7. Data Visualization

# In[7]:


import matplotlib.pyplot as plt
import seaborn as sns

# Plot the number of medals won by each country
plt.figure(figsize=(10, 6))
sns.barplot(x='Medal Count', y='Region', data=medals_by_region.head(10))
plt.title('Top 10 Countries by Medal Count')
plt.xlabel('Medal Count')
plt.ylabel('Country')
plt.show()


# # 8. Advanced Filtering

# In[8]:


# Find athletes who competed in multiple sports
multi_sport_athletes = merged_df.groupby('Name').filter(lambda x: x['Sport'].nunique() > 1)
multi_sport_athletes.head()


# # 9. Pivot Tables

# In[9]:


# Create a pivot table of medals by year and sport
pivot_table = pd.pivot_table(merged_df, values='Medal', index='Year', columns='Sport', aggfunc='count', fill_value=0)
pivot_table.head()


# # 10. Applying Functions

# In[10]:


# Define a function to categorize athletes by age
def age_category(age):
    if age < 20:
        return 'Teen'
    elif age < 30:
        return 'Adult'
    else:
        return 'Senior'

# Apply the function to the 'Age' column
merged_df['Age Category'] = merged_df['Age'].apply(age_category)
print("DataFrame with age categories:")
print(merged_df[['Name', 'Age', 'Age Category']].head())


# # 11.Groupby

# In[11]:


# Group by 'Year' and calculate the mean age
mean_age_by_year = merged_df.groupby('Year')['Age'].mean()
print(mean_age_by_year)


# # 12.Apply

# In[18]:


# Apply a function to calculate the Body Mass Index (BMI)
merged_df['BMI'] = merged_df.apply(lambda row: row['Weight'] / (row['Height']/100)**2 if row['Height'] > 0 else None, axis=1)
merged_df[['Name', 'Height', 'Weight', 'BMI']].head()


# # 13.map() and applymap()

# In[17]:


# Map a dictionary to a Series to convert medal names to numerical values
medal_map = {'Gold': 3, 'Silver': 2, 'Bronze': 1}
merged_df['Medal_Value'] = merged_df['Medal'].map(medal_map)
merged_df[['Name', 'Medal', 'Medal_Value']].head()


# # 14.Astype()

# In[14]:


# Convert 'Year' column to string
merged_df['Year'] = merged_df['Year'].astype(str)
print(merged_df['Year'].dtype)


# # 15.drop_duplicates()

# In[16]:


# Drop duplicate rows based on 'Name' and 'Year'
unique_athletes = merged_df.drop_duplicates(subset=['Name', 'Year'])
unique_athletes.head()


# In[ ]:




