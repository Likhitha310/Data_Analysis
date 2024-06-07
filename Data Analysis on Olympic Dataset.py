#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re


# # Reading the dataset

# In[2]:


# Load the datasets
athlete_events_df = pd.read_csv('athlete_events.csv')
noc_regions_df = pd.read_csv('noc_regions.csv')


# # Initial Exploration

# In[3]:


athlete_events_df.head()


# In[4]:


athlete_events_df.info()


# In[5]:


athlete_events_df.describe()


# In[6]:


noc_regions_df.head()


# In[7]:


noc_regions_df.info()


# In[8]:


noc_regions_df.describe()


# # Data Cleaning

# In[9]:


athlete_events_df.isnull().sum()
noc_regions_df.isnull().sum()
# Handling missing values in athlete_events_df
athlete_events_df = athlete_events_df.dropna(subset=['Age', 'Height', 'Weight'])  # Example action
athlete_events_df


# # Merging Datasets

# In[10]:


# Merge datasets on 'NOC'
merged_df = pd.merge(athlete_events_df, noc_regions_df, how='left', left_on='NOC', right_on='NOC')


# # Data Analysis

# In[11]:


#Descriptive Statistics
# Basic statistics about the number of medals
medal_counts = merged_df['Medal'].value_counts()
print(medal_counts)


# Data Visualization

# In[12]:


# Plot the distribution of medals
sns.countplot(x='Medal', data=merged_df)
plt.title('Distribution of Medals')
plt.show()



# In[13]:


# Plot the number of medals won by different countries
country_medals = merged_df.groupby('region')['Medal'].count().sort_values(ascending=False)
country_medals.head(10).plot(kind='bar')
plt.title('Top 10 Countries by Medal Count')
plt.show()


# In[14]:


# Plot the number of medals won over the years
yearly_medals = merged_df.groupby('Year')['Medal'].count()
yearly_medals.plot(kind='line')
plt.title('Number of Medals Won Over the Years')
plt.show()


# # Advanced Analysis

# In[15]:


#Correlation Analysis
# Analyzing the correlation between numerical variables
correlation_matrix = merged_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


# In[16]:


#Identifying Trends and Patterns
# Analyzing trends in medal wins by gender over time
gender_medals = merged_df.groupby(['Year', 'Sex'])['Medal'].count().unstack()
gender_medals.plot(kind='line')
plt.title('Medals Won by Gender Over the Years')
plt.show()


# # Exploratory Data Analysis (EDA)

# In[17]:


#Interactive Visualizations
import plotly.express as px

# Interactive plot of medal distribution by country
fig = px.bar(country_medals.head(10), title='Top 10 Countries by Medal Count')
fig.show()


# In[18]:


#Analyze the distribution of athletes' physical characteristics.
# Distribution of Age
sns.histplot(merged_df['Age'], bins=30, kde=True)
plt.title('Distribution of Age')
plt.show()


# In[19]:


# Distribution of Height
sns.histplot(merged_df['Height'], bins=30, kde=True)
plt.title('Distribution of Height')
plt.show()


# In[20]:


# Distribution of Weight
sns.histplot(merged_df['Weight'], bins=30, kde=True)
plt.title('Distribution of Weight')
plt.show()


# # Time Series Analysis

# In[21]:


# Time series analysis of medals won over the years
yearly_medals = merged_df.groupby('Year')['Medal'].count()
yearly_medals.plot(kind='line')
plt.title('Number of Medals Won Over the Years')
plt.xlabel('Year')
plt.ylabel('Number of Medals')
plt.show()


# # Geospatial Analysis

# In[22]:


import geopandas as gpd

# Load a world map
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

# Merge with the medal data
country_medals = merged_df.groupby('region')['Medal'].count().reset_index()
world_medals = world.merge(country_medals, how='left', left_on='name', right_on='region')

# Plot the world map with medal counts
world_medals.plot(column='Medal', cmap='OrRd', legend=True,
                  legend_kwds={'label': "Number of Medals",
                               'orientation': "horizontal"})
plt.title('Medal Distribution by Country')
plt.show()


# # Athlete Performance Analysis

# In[23]:


# Top athletes by number of medals won
top_athletes = merged_df[merged_df['Medal'].notnull()].groupby('Name')['Medal'].count().sort_values(ascending=False).head(10)
print(top_athletes)
top_athletes.plot(kind='bar')
plt.title('Top 10 Athletes by Medal Count')
plt.show()


# # Sports-Specific Analysis

# In[24]:


# Medals won in different sports
sports_medals = merged_df.groupby('Sport')['Medal'].count().sort_values(ascending=False).head(10)
print(sports_medals)
sports_medals.plot(kind='bar')
plt.title('Top 10 Sports by Medal Count')
plt.show()


# # Gender Analysis

# In[25]:


# Number of medals won by gender over the years
gender_medals = merged_df[merged_df['Medal'].notnull()].groupby(['Year', 'Sex'])['Medal'].count().unstack()
gender_medals.plot(kind='line')
plt.title('Medals Won by Gender Over the Years')
plt.xlabel('Year')
plt.ylabel('Number of Medals')
plt.show()


# # Machine Learning

# In[26]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


# In[27]:


# Prepare data for modeling
features = merged_df[['Age', 'Height', 'Weight']]
features = features.dropna()
target = merged_df['Medal'].notnull().astype(int)  # Binary target: 1 if won a medal, 0 otherwise
target = target[features.index]


# In[28]:


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)


# In[29]:


# Train a Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)


# In[30]:


# Make predictions and evaluate the model
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))


# # Clustering Athletes

# In[31]:


from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# In[32]:


# Scale the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)


# In[33]:


# Perform K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(scaled_features)


# In[34]:


# Add cluster labels to the DataFrame
features['Cluster'] = clusters


# In[35]:


# Visualize the clusters
sns.scatterplot(x='Height', y='Weight', hue='Cluster', data=features, palette='viridis')
plt.title('Clustering Athletes by Height and Weight')
plt.show()


# # Feature Engineering

# In[36]:


# Calculate BMI
merged_df['BMI'] = merged_df['Weight'] / (merged_df['Height']/100)**2

# Analyze BMI distribution
sns.histplot(merged_df['BMI'].dropna(), bins=30, kde=True)
plt.title('Distribution of BMI')
plt.show()


# # Comparative Analysis

# In[37]:


# Compare the number of medals won in Summer vs Winter Olympics
season_medals = merged_df.groupby('Season')['Medal'].count()
season_medals.plot(kind='bar')
plt.title('Number of Medals Won: Summer vs Winter Olympics')
plt.show()


# # MACHINE LEARNING ALGORITHMS

# Anomaly Detection with Isolation Forest

# In[38]:


from sklearn.ensemble import IsolationForest


# In[39]:


# Prepare data
features = merged_df[['Age', 'Height', 'Weight']]
features = features.dropna()


# In[40]:


# Train Isolation Forest
clf = IsolationForest(random_state=42)
clf.fit(features)


# In[41]:


# Predict anomalies
anomalies = clf.predict(features)
features['Anomaly'] = anomalies


# In[42]:


# Visualize anomalies
sns.scatterplot(x='Height', y='Weight', hue='Anomaly', data=features, palette='coolwarm')
plt.title('Anomaly Detection in Athletes')
plt.show()


#  Deep Learning with PyTorch

# In[43]:


import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler


# In[44]:


# Prepare data
features = merged_df[['Age', 'Height', 'Weight']]
features = features.dropna()
target = merged_df['Medal'].notnull().astype(int)  # Binary target: 1 if won a medal, 0 otherwise
target = target[features.index]


# In[45]:


# Scale features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)


# In[46]:


# Convert to PyTorch tensors
X = torch.tensor(scaled_features, dtype=torch.float32)
y = torch.tensor(target.values, dtype=torch.float32)


# In[47]:


# Define the neural network model
class MedalPredictor(nn.Module):
    def __init__(self):
        super(MedalPredictor, self).__init__()
        self.fc1 = nn.Linear(3, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


# In[48]:


# Initialize the model, loss function, and optimizer
model = MedalPredictor()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# In[49]:


# Train the model
epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X).squeeze()
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')


# In[50]:


# Evaluate the model
model.eval()
with torch.no_grad():
    predictions = model(X).squeeze()
    predicted_classes = (predictions > 0.5).int()
    accuracy = (predicted_classes == y.int()).float().mean()
    print(f'Accuracy: {accuracy.item():.4f}')


# Graph-Based Analysis with NetworkX

# In[51]:


import networkx as nx


# In[52]:


# Create a graph where nodes are athletes and edges are shared events
G = nx.Graph()


# In[53]:


# Add nodes for each athlete
athletes = merged_df['Name'].unique()
G.add_nodes_from(athletes)


# In[ ]:


# Add edges for shared events
for _, event in merged_df.groupby('Event'):
    participants = event['Name'].unique()
    for i in range(len(participants)):
        for j in range(i + 1, len(participants)):
            G.add_edge(participants[i], participants[j])


# In[ ]:


# Analyze the graph
degree_centrality = nx.degree_centrality(G)
print(sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10])


# In[ ]:


# Visualize the graph
pos = nx.spring_layout(G)
plt.figure(figsize=(12, 12))
nx.draw_networkx_nodes(G, pos, node_size=20)
nx.draw_networkx_edges(G, pos, alpha=0.1)
plt.title('Graph of Athletes Connected by Shared Events')
plt.show()

