#!/usr/bin/env python
# coding: utf-8

# In[1]:


from bs4 import BeautifulSoup
import requests
import pandas as pd
import time
import re


# In[2]:


pages = [1,2,3,4,5]

a = []
Exp = []
Price = []
Rating = []



for page in pages:
  url = "https://www.lybrate.com/pune/psychiatrist?page={}".format(page)
  resp = requests.get(url)
  soup = BeautifulSoup(resp.content, 'lxml')

  for i in soup.findAll('a', {'itemprop': 'url'}):
    a.append(i.get_text(strip=True))

  for i in soup.findAll('span', {'class': 'lybText--green'}):
    Exp.append(i.get_text(strip=True))

  for i in soup.findAll('div', {'span': ''}):
    Price.append(i.get_text(strip=True))

  for i in soup.findAll('div', {'class': 'grid__col-xs-10 grid--direction-row'}):
    Rating.append(i.get_text(strip=True))


# In[3]:


Doc_name = [a[i] for i in range(int(len(a))) if i%2==0]
Hospital_name = [a[i] for i in range(int(len(a))) if i%2!=0]


# In[4]:


Price[30]
Price[63]
Price[94]


# In[5]:


arr=[]
for i in Rating:
  if re.findall(r"₹\w",i):
    arr.append(i)


# In[6]:


len(arr)


# In[11]:


url = "https://www.thehindu.com/"
resp = requests.get(url)
soup = BeautifulSoup(resp.content, 'html')


# In[12]:


andi = []
for i in soup.findAll('div', {'class': 'listing-doctor-card'}):
    andi.append(i.get_text(strip=True))


# In[13]:


soup


# In[ ]:




