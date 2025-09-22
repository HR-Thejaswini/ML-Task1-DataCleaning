#!/usr/bin/env python
# coding: utf-8

# In[7]:


# -------------------------
# Task 1: Data Cleaning & Preprocessing
# -------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# 1. Import Dataset & Explore
df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")

print("🔹 First 5 rows of dataset:")
print(df.head())

print("\n🔹 Dataset Info:")
print(df.info())

print("\n🔹 Missing Values:")
print(df.isnull().sum())

# 2. Handle Missing Values
# Fill missing Age with median
df['Age'].fillna(df['Age'].median(), inplace=True)

# Fill missing Embarked with mode
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop Cabin (too many missing values)
if 'Cabin' in df.columns:
    df.drop(columns=['Cabin'], inplace=True)

print("\n🔹 Missing values after handling:")
print(df.isnull().sum())

# 3. Encode Categorical Variables
# Label Encode 'Sex'
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# One-Hot Encode 'Embarked'
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

print("\n🔹 After Encoding:")
print(df.head())

# 4. Normalize / Standardize Numerical Features
scaler = StandardScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])

print("\n🔹 After Scaling:")
print(df[['Age', 'Fare']].head())

# 5. Detect & Remove Outliers
# Boxplot before removing outliers
plt.figure(figsize=(6,4))
sns.boxplot(x=df['Fare'])
plt.title("Boxplot of Fare (Before Outlier Removal)")
plt.show()

# Remove outliers using IQR method
Q1 = df['Fare'].quantile(0.25)
Q3 = df['Fare'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['Fare'] >= Q1 - 1.5*IQR) & (df['Fare'] <= Q3 + 1.5*IQR)]

# Boxplot after removing outliers
plt.figure(figsize=(6,4))
sns.boxplot(x=df['Fare'])
plt.title("Boxplot of Fare (After Outlier Removal)")
plt.show()

print("\n🔹 Final Cleaned Dataset Shape:", df.shape)
print(df.head())


# In[ ]:




