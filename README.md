# Data-Science-Process

# AIM:

     To Perform Data Science process on a complex dataset and save the data to a file.

# EXPLANATION:

     The Data Science Process is a systematic approach to solving data-related problems and consists of the following steps:
     
        1.	Problem Definition
     
        2.	Data Collection
        
        3.	Data Exploration
        
        4.	Data Modelling
        
        5.	Evaluation
        
        6.	Deployment
        
        7.	Monitoring and Maintenance
        
# ALGORITHM:

      Step 1: Read the given data.
      
      Step 2: Clean the Data Set using Data Cleaning Process.
      
      Step 3: Apply Feature Generation/Feature selection techniques on the data set.
      
      Step 4: Apply EDA/Data visualization techniques to all the features of the data.

# CODE:

import numpy as np 

import pandas as pd

import os

import seaborn as sns

import matplotlib.pyplot as plt

df = pd.read_csv("/content/Disease_symptom_and_patient_profile_dataset.csv")

df

df.head()

df.info()

df.tail()

df.isnull().sum()

df.shape

df.nunique()

from sklearn import datasets

iris = datasets.load_iris()

X = iris.data[:, [0, 2]]

Y = iris.target

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1)

from sklearn.preprocessing import MinMaxScaler, StandardScaler

data = [[10, 20, 30],

       [5, 15, 25],

       [3, 6, 9],

       [8, 12, 16]]

scaler = MinMaxScaler()

scaled_data = scaler.fit_transform(data)

print("Min-max scaled data:")

print(scaled_data)

scaler = StandardScaler()

scaled_data = scaler.fit_transform(data)

print("Standard scaled data:")

print(scaled_data)

np.random.seed(42)

data = np.random.normal(0, 1, 1000)

plt.figure(figsize=(5, 5))

plt.hist(data, bins=30, edgecolor='black')

plt.xlabel('Value')

plt.ylabel('Frequency')

plt.title('Histogram')

plt.show()

sns.catplot(x='Outcome Variable',y='Age',data=df,kind="swarm")

sns.catplot(x='Difficulty Breathing' , kind='count',data=df , hue = "Outcome Variable")

sns.catplot(x='Fatigue' , kind='count',data=df , hue = "Outcome Variable")

plt.figure(figsize=(5,5))

sns.barplot(x='Fever',y='Age',data =df);

sns.displot(df['Age'],kde=True)

df.groupby('Gender').size().plot(kind='pie', autopct='%.2f')

sns.catplot(x='Cough' , kind='count',data=df , hue = "Cholesterol Level")

df.groupby('Blood Pressure').size().plot(kind='pie', autopct='%.2f')

sns.catplot(x='Gender' , kind='count',data=df , hue = "Cholesterol Level")

df = df.iloc[:,1:]

df.groupby('Fatigue').size().plot(kind='pie', autopct='%.2f')

# OUTPUT:

![image](https://github.com/akshitha-ks/Data-Science-Process/assets/123535064/cf34c8f1-92cc-49b2-b1cd-188707d8237a)

![image](https://github.com/akshitha-ks/Data-Science-Process/assets/123535064/9b249640-5cd1-43d5-9452-1ac824c88ad0)

![image](https://github.com/akshitha-ks/Data-Science-Process/assets/123535064/26e0c58b-957c-4e5c-87de-84c606ae1138)

![image](https://github.com/akshitha-ks/Data-Science-Process/assets/123535064/04a0be3e-a9a3-4e39-aece-53e0556dbe88)

![image](https://github.com/akshitha-ks/Data-Science-Process/assets/123535064/b4c3a61e-c0e4-4767-b3fa-29409fb04be5)

![image](https://github.com/akshitha-ks/Data-Science-Process/assets/123535064/f9e979d1-e14b-4ddc-87a7-ba70667395c2)

![image](https://github.com/akshitha-ks/Data-Science-Process/assets/123535064/09bf97cc-bacf-4984-92c0-9fc61e0811fd)

![image](https://github.com/akshitha-ks/Data-Science-Process/assets/123535064/1d953a4c-d024-4c78-bc2a-d4b6fab1484c)

![image](https://github.com/akshitha-ks/Data-Science-Process/assets/123535064/11bb05d5-87b7-4d1e-8361-1771ae37d4fa)

![image](https://github.com/akshitha-ks/Data-Science-Process/assets/123535064/d453879a-40dc-4f82-bda3-80b8f0250ff8)

![image](https://github.com/akshitha-ks/Data-Science-Process/assets/123535064/0c3283d1-13d0-449d-82ce-82a058e50eb4)

![image](https://github.com/akshitha-ks/Data-Science-Process/assets/123535064/f9eb1e0a-dd22-4e19-999d-5b49927bc568)

![image](https://github.com/akshitha-ks/Data-Science-Process/assets/123535064/e22984fd-2bf1-44b6-99ed-fc443709049d)

![image](https://github.com/akshitha-ks/Data-Science-Process/assets/123535064/0ae5fc44-bf47-496a-9a6b-a48a117f8920)

![image](https://github.com/akshitha-ks/Data-Science-Process/assets/123535064/99ef5b39-39be-44d1-bbad-73c642b44c76)

![image](https://github.com/akshitha-ks/Data-Science-Process/assets/123535064/49c71f95-d588-475e-9aba-82557950562c)

![image](https://github.com/akshitha-ks/Data-Science-Process/assets/123535064/86896989-0e01-401b-b61d-fb48beb1d31c)

![image](https://github.com/akshitha-ks/Data-Science-Process/assets/123535064/de85a426-1991-47f1-9dd2-a8a9530f4453)

![image](https://github.com/akshitha-ks/Data-Science-Process/assets/123535064/cf93ffd3-cd5a-407a-926a-bf905bc0e660)


# RESULT:

      Thus, the Data Science process for the given datasets had been executed successfully.
