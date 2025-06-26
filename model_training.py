import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

from sklearn.impute import SimpleImputer

from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline, make_pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


#read the file
data = pd.read_csv('synthetic_health_lifestyle_dataset.csv')


############################
#Step1. data exploring (EDA)
#############################
print(data.head(10), '\n')
print(data.shape, '\n')
#(7500, 13)
print(data.describe, '\n')
print(data.describe(include = ['O']), '\n')
'''
       Gender Smoker Exercise_Freq Diet_Quality Alcohol_Consumption Chronic_Disease
count    7500   7500          5621         7500                5608            7500
unique      3      2             3            4                   3               2
top      Male     No         Daily         Good                 Low              No
freq     2551   5263          1925         1918                1893            6052
'''

print(data.dtypes, '\n')
'''
ID                       int64
Age                      int64
Gender                  object
Height_cm              float64
Weight_kg              float64
BMI                    float64
Smoker                  object
Exercise_Freq           object
Diet_Quality            object
Alcohol_Consumption     object
Chronic_Disease         object
Stress_Level             int64
Sleep_Hours            float64
dtype: object
'''

# Drop the unnecessary column "ID"
df = data.drop(columns = ['ID'])

# Check Numerical data distribution by = 'Chronic_Disease'
# Age, Height_cm, Weight_kg, BMI, Stress_Level, Sleep_Hours
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12,7))

#Age boxplot
sns.boxplot(df, x='Chronic_Disease', y='Age', ax = axes[0,0])
axes[0,0].set_title('Age distribution by Chronic_Disease')
axes[0,0].set_xlabel('Chronic_Disease')
axes[0,0].set_ylabel('Age')


# Height_cm boxplot
sns.boxplot(df, x='Chronic_Disease', y='Height_cm', ax = axes[0,1])
axes[0,1].set_title('Height_cm distribution by Chronic_Disease')
axes[0,1].set_xlabel('Chronic_Disease')
axes[0,1].set_ylabel('Height_cm')

#Weight_kg boxplot
sns.boxplot(df, x='Chronic_Disease', y='Weight_kg', ax = axes[0,2])
axes[0,2].set_title('Weight_kg distribution by Chronic_Disease')
axes[0,2].set_xlabel('Chronic_Disease')
axes[0,2].set_ylabel('Weight_kg')

#BMI boxplot
sns.boxplot(df, x='Chronic_Disease', y='BMI', ax = axes[1,0])
axes[1,0].set_title('BMI distribution by Chronic_Disease')
axes[1,0].set_xlabel('Chronic_Disease')
axes[1,0].set_ylabel('BMI')

#Stress level
sns.boxplot(df, x='Chronic_Disease', y='Stress_Level', ax = axes[1,1])
axes[1,1].set_title('Stress_Level by Chronic_Disease')
axes[1,1].set_xlabel('Chronic_Disease')
axes[1,1].set_ylabel('Stress_Level')

#Sleep_Hours
sns.boxplot(df, x='Chronic_Disease', y='Sleep_Hours', ax = axes[1,2])
axes[1,2].set_title('Sleep_Hours distribution by Chronic_Disease')
axes[1,2].set_xlabel('Chronic_Disease')
axes[1,2].set_ylabel('Sleep_Hours')

plt.tight_layout()
plt.show()


# Check Categorical data distribution by = 'Chronic_Disease'
# Gender, Smoker, Exercise_Freq, Diet_Quality, Alcohol_Consumption, Chronic_Disease 
df_Gender = df.groupby(['Chronic_Disease','Gender']).size().reset_index(name = 'count')
sns.barplot(df_Gender, x='Chronic_Disease', y='count', hue='Gender')
plt.title('Count of Gender by Chronic Disease Status')
plt.xlabel('Chronic Disease')
plt.ylabel('Count')

plt.show()