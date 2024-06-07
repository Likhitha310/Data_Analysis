#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Dataset:https://raw.githubusercontent.com/sahilrahman12/Technology-Lookup-Web-Application/main/technologies.json


# In[2]:


import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np
import requests


# In[3]:


url = "https://github.com/ozlerhakan/mongodb-json-files/raw/master/datasets/restaurant.json"

# Download the JSON file locally
filename = "restaurant.json"
response = requests.get(url)
with open(filename, "wb") as f:
    f.write(response.content)

# Read the JSON file using Pandas
df = pd.read_json(filename, lines=True)


# In[4]:


df.head()


# In[5]:


df.shape


# In[6]:


df.columns


# In[7]:


df.info()


# In[8]:


df.describe()


# # Accessing data from specific columns:

# In[10]:


# Accessing data from the 'name' column
names = df['name']
names


# In[12]:


# Accessing data from the 'rating' column
ratings = df['rating']
ratings


# In[13]:


# Accessing data from the 'type_of_food' column
food_types = df['type_of_food']
food_types


# # Performing operations on specific columns:

# In[18]:


# Convert non-numeric values in the 'rating' column to NaN
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')


# In[19]:


# Calculating the mean rating
mean_rating = df['rating'].mean()
print("Mean Rating:", mean_rating)



# In[15]:


# Counting the number of unique food types
num_food_types = df['type_of_food'].nunique()
num_food_types


# In[20]:


# Filtering the DataFrame based on a condition in the 'rating' column
highly_rated_restaurants = df[df['rating'] > 4.5]
print("Highly Rated Restaurants:")
print(highly_rated_restaurants)



# In[21]:


# Grouping the DataFrame by food type and calculating the mean rating for each type
mean_rating_by_food_type = df.groupby('type_of_food')['rating'].mean()
print("Mean Rating by Food Type:")
print(mean_rating_by_food_type)


# # Counting the number of unique food types

# In[22]:


num_food_types = df['type_of_food'].nunique()
print("Number of Unique Food Types:", num_food_types)


# # Counting the frequency of each food type

# In[23]:


food_type_counts = df['type_of_food'].value_counts()
print("Frequency of Each Food Type:")
print(food_type_counts)


# # Filtering the DataFrame based on a condition in the 'type_of_food' column

# In[24]:


chinese_restaurants = df[df['type_of_food'] == 'Chinese']
print("Chinese Restaurants:")
print(chinese_restaurants)


# # Calculating summary statistics for the 'rating' column

# In[25]:


rating_summary = df['rating'].describe()
print("Summary Statistics for Rating:")
print(rating_summary)


# # Visualizing the distribution of ratings

# In[26]:


plt.hist(df['rating'], bins=10, color='skyblue', edgecolor='black')
plt.title('Distribution of Ratings')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.show()


# # Arrays

# In[28]:


# 1. Arrays
# Accessing a specific row or column
first_row = df.iloc[0]  # Accessing the first row
name_column = df['name']  # Accessing the 'name' column
name_column


# # Linked Lists

# In[29]:


class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

# Creating linked list nodes from the 'name' column
head = None
prev_node = None
for name in df['name']:
    new_node = Node(name)
    if head is None:
        head = new_node
    if prev_node:
        prev_node.next = new_node
    prev_node = new_node


# In[30]:


# Traversing the linked list
current_node = head
while current_node:
    print(current_node.data)
    current_node = current_node.next


# # Queues

# In[31]:


class Queue:
    def __init__(self):
        self.items = []

    def enqueue(self, item):
        self.items.append(item)

    def dequeue(self):
        return self.items.pop(0) if not self.is_empty() else None

    def is_empty(self):
        return len(self.items) == 0


# In[32]:


# Creating a queue with restaurant names
restaurant_queue = Queue()
for name in df['name']:
    restaurant_queue.enqueue(name)


# In[33]:


# Processing each restaurant name in FIFO order
while not restaurant_queue.is_empty():
    name = restaurant_queue.dequeue()
    print("Processing:", name)


# In[ ]:





# In[ ]:




