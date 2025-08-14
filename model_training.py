###############################################################
# Project Description:
#
# This project applies machine learning (ML) techniques for 
# classification using the Sleep Health and Lifestyle Dataset 
# from Kaggle. 
# By analyzing body parameters and lifestyle factors, the goal 
# is to build a predictive model for identifying the presence 
# of chronic disease.
###############################################################

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold

from sklearn.preprocessing import StandardScaler, RobustScaler, OrdinalEncoder, OneHotEncoder, FunctionTransformer

from sklearn.impute import SimpleImputer

from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline, make_pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score



#The file was downloaded from Kaggle: "Health And Lifestyle Dataset".
#https://www.kaggle.com/datasets/sahilislam007/health-and-lifestyle-dataset/code
#read the file
data = pd.read_csv('synthetic_health_lifestyle_dataset.csv')


############################
#Step1. data exploring (EDA)
#############################
print(data.head(10), '\n')

print(data.isna().sum())
# Check NA value in all columns
# 'Exercise_Freq' and 'Alcohol_Consumption' columns need to check NA value
'''
ID                        0
Age                       0
Gender                    0
Height_cm                 0
Weight_kg                 0
BMI                       0
Smoker                    0
Exercise_Freq          1879
Diet_Quality              0
Alcohol_Consumption    1892
Chronic_Disease           0
Stress_Level              0
Sleep_Hours               0
dtype: int64
(7500, 13)
'''
Exercise_Freq_unique = data['Exercise_Freq'].unique()
print(f'Exercise_Freq unique: {Exercise_Freq_unique}')
#Exercise_Freq unique: [nan '1-2 times/week' 'Daily' '3-5 times/week']

Alcohol_Consumption_unique = data['Alcohol_Consumption'].unique()
print(f'Alcohol_Consumption unique: {Alcohol_Consumption_unique}')
#Alcohol_Consumption unique: [nan 'High' 'Moderate' 'Low']

# Actually, NA value in both columns mean 'no', not really a missing value
# Replace all NA values with 'no'
data['Exercise_Freq'] = data['Exercise_Freq'].fillna('no')
data['Alcohol_Consumption'] = data['Alcohol_Consumption'].fillna('no')


print(data.shape, '\n')
#(7500, 13)
print(data.describe, '\n')
print(data.describe(include = ['O']), '\n')
'''
       Gender Smoker Exercise_Freq Diet_Quality Alcohol_Consumption Chronic_Disease
count    7500   7500          7500         7500                7500            7500
unique      3      2             4            4                   4               2
top      Male     No         Daily         Good                 Low              No
freq     2551   5263          1925         1918                1893            6052
'''

#check all the unique values in all objective columns
for c in data.columns:
    if data[c].dtypes == 'object':
        print(f'{c} unique values:{data[c].unique()}')
#Gender unique values:['Other' 'Female' 'Male']
#Smoker unique values:['Yes' 'No']
#Exercise_Freq unique values:['no' '1-2 times/week' 'Daily' '3-5 times/week']
#Diet_Quality unique values:['Poor' 'Good' 'Excellent' 'Average']
#Alcohol_Consumption unique values:['no' 'High' 'Moderate' 'Low']
#Chronic_Disease unique values:['No' 'Yes']


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

#check target 'Chronic_Disease' distribution
print(data['Chronic_Disease'].value_counts())
print(data['Chronic_Disease'].value_counts(normalize = True))
# No     6052
# Yes    1448
# No     0.806933
# Yes    0.193067

count_Chronic = data['Chronic_Disease'].value_counts().reset_index()
count_Chronic.columns = ['Chronic_Disease','counts']
print(count_Chronic)


#draw a barplot for Count of Chronic_Disease
ax = sns.barplot(data = count_Chronic, x = 'Chronic_Disease', y = 'counts', hue = 'Chronic_Disease')
ax.set_title('Count of Chronic_Disease')
ax.set_xlabel('Chronic_Disease')
ax.set_ylabel('counts')
plt.savefig('Count of Chronic_Disease_barplot.png', dpi = 300)
#plt.show()
plt.close()
input()
#The target variable 'Chronic_Disease' has an imbalanced class distribution
#use 'stratify' to split data
#use  class_weight='balanced' whem training models


# Drop the unnecessary column "ID"
df = data.drop(columns = ['ID'])

# Use boxplot to check data distribution to check Numerical data distribution by = 'Chronic_Disease'
num_col = []
for c in df.columns:
    if df[c].dtypes !=  'object':
        num_col.append(c)
        print(f'numerical col: {c}')
# numerical col: Age
# numerical col: Height_cm
# numerical col: Weight_kg
# numerical col: BMI
# numerical col: Stress_Level
# numerical col: Sleep_Hours

fig1, axes1 = plt.subplots(nrows=2, ncols=3, figsize=(14,7))
axes1 = axes1.flatten()

for i, numcol in enumerate(num_col):
    sns.boxplot(df, x = 'Chronic_Disease', y = numcol, ax = axes1[i], hue = 'Chronic_Disease', palette='Set2')
    axes1[i].set_title(f'{numcol} distribution by Chronic_Disease')
    axes1[i].set_xlabel('Chronic_Disease')
    axes1[i].set_ylabel(f'{numcol}')

plt.tight_layout()
plt.savefig('Numerical_data_boxplot.png', dpi=300)
#plt.show()
plt.close()
   


# Use barplot to check Categorical data distribution by = 'Chronic_Disease'
catcol = []
for c in df.columns:
    if df[c].dtypes == 'object':
        if c != 'Chronic_Disease':
            catcol.append(c)
            print(f'categorical col: {c}')
# categorical col: Gender
# categorical col: Smoker
# categorical col: Exercise_Freq
# categorical col: Diet_Quality
# categorical col: Alcohol_Consumption
# categorical col: Chronic_Disease


# Gender, Smoker, Exercise_Freq, Diet_Quality, Alcohol_Consumption
fig2, axes2 = plt.subplots(nrows=2, ncols=3, figsize=(14,8))
axes2 = axes2.flatten()

for i, col in enumerate(catcol):
    temp_df = df.groupby(['Chronic_Disease', col]).size().reset_index(name = 'count')
    sns.barplot(temp_df, x = 'Chronic_Disease', y = 'count', hue = col, ax = axes2[i])
    axes2[i].set_title(f'Count of {col} by Chronic Disease Status')
    axes2[i].set_xlabel('Chronic Disease')
    axes2[i].set_ylabel('Count')

#close empty figure
for ax in axes2[len(catcol):]:
    ax.axis('off')

plt.tight_layout()
plt.savefig('Categorical_data_barplot.png', dpi = 300)
#plt.show()
plt.close()

#Correlation Heatmap
cor = df.corr(numeric_only=True)
plt.figure(figsize=(14,10))

sns.heatmap(
    data = cor,
    annot = True,
    fmt = ".2f",
    linewidths = 0.5,
    annot_kws = {"size": 14, "color": "black"},
    cmap ='coolwarm',
    vmin = -1, vmax = 1,
    center=0
)
plt.title("Correlation Heatmap", size = 16)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.savefig('Correlation Heatmap.png', dpi = 300)
plt.show()

#input()

#########################################################################################
# >>> EDA results evulation
#
# The target variable 'Chronic_Disease' has an imbalanced class distribution:
# No     0.806933
# Yes    0.193067
# use 'stratify' to split data
# use class_weight = 'balanced' whem training models
#
# Based on the boxplot of the numerical data, different standization methods must be used
# Columns with many outliers: BMI, Sleep_Hours --> RobustScaler
# Columns without outliers: Age, Stress_Level --> StandardScaler
#
# Based on the barplot of the Categorical data
# Only 'Smoker' data show some relationship with Chronic_Disease
# But still can put all the columns in model training
# Gender--> OneHotEncoder
# Smoker, Exercise_Freq, Diet_Quality, Alcohol_Consumption --> OrdinalEncoder
#
# Base on the heatmap result
# BMI has strong corrlection between Height and Weight
# It make sense, beacuse BMI was calculated from the formula: Weight(kg)/(Height(m))**2
# In order to reduce the 'Multicollinearity'
# Drop the columns 'Height' and 'Weight' but keep 'BMI'
#########################################################################################



############################
#Step2. data transformation
#############################

#data split
#drop columns 'Height_cm','Weight_kg'
# use 'stratify' when split data
X = df.drop(columns = ['Chronic_Disease','Height_cm','Weight_kg'])
y = df['Chronic_Disease'].map({'No':0 ,'Yes':1})
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size = 0.2, random_state = 1)

#columns for data transformation
RobustScaler_columns = ['BMI','Sleep_Hours']
StandardScaler_columns = ['Age','Stress_Level']
one_hot_columns = ['Gender']

Smoker_rank = ['No','Yes']
Exercise_Freq_rank = ['no','1-2 times/week','3-5 times/week','Daily']
Diet_Quality_rank = ['Poor','Average','Good','Excellent']
Alcohol_Consumption_rank = ['no','Low','Moderate','High']


mct = make_column_transformer(
    (RobustScaler(), RobustScaler_columns),
    (StandardScaler(), StandardScaler_columns),
    (OneHotEncoder(handle_unknown = 'ignore'), one_hot_columns),
    (Pipeline([
        ('ordinal', OrdinalEncoder(categories=[Smoker_rank])),
        ('scale', StandardScaler())
    ]), ['Smoker']),

    (Pipeline([
        ('ordinal', OrdinalEncoder(categories=[Exercise_Freq_rank])),
        ('scale', StandardScaler())
    ]), ['Exercise_Freq']),

    (Pipeline([
        ('ordinal', OrdinalEncoder(categories=[Diet_Quality_rank])),
        ('scale', StandardScaler())
    ]), ['Diet_Quality']),

    (Pipeline([
        ('ordinal', OrdinalEncoder(categories=[Alcohol_Consumption_rank])),
        ('scale', StandardScaler())
    ]), ['Alcohol_Consumption']),
    remainder = 'drop',
    n_jobs= -1
)



############################
#Step3. model training
#
# Add class_weight = 'balanced' in model if can
# Use StratifiedKFold() to split data  
#############################
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

#LogisticRegression
lr = LogisticRegression()
lr_CV_pipeline = make_pipeline(mct, lr)

cv = StratifiedKFold(n_splits = 3, shuffle = True, random_state = 1)

lr_param_grid = {
    'logisticregression__class_weight': [None, 'balanced'],
    'logisticregression__penalty':['l2'],
    'logisticregression__C':[0.01, 0.1, 1, 10],
    'logisticregression__max_iter':[100, 500, 1000]
}

lr_CV = GridSearchCV(lr_CV_pipeline, lr_param_grid, cv = cv, n_jobs= -1)
lr_CV.fit(X_train, y_train)
y_pred_lr = lr_CV.predict(X_test)

print('> LogisticRegression results')
print(classification_report(y_test,y_pred_lr))
print(confusion_matrix(y_test,y_pred_lr))
print(f'Best params: {lr_CV.best_params_}', '\n')


#SVC
svc = SVC()
svc_CV_piprline = make_pipeline(mct, svc)

cv = StratifiedKFold(n_splits = 3, shuffle = True, random_state = 1)

svc_param_grid = {
    'svc__C': [0.1, 1, 10],
    'svc__kernel': ['linear', 'rbf', 'poly'],
    'svc__gamma': ['scale', 'auto'],
    'svc__degree': [2, 3, 4],
    'svc__class_weight': [None, 'balanced']
}

svc_CV = GridSearchCV(svc_CV_piprline, svc_param_grid, cv = cv, n_jobs = -1)

svc_CV.fit(X_train, y_train)
y_pred_svc = svc_CV.predict(X_test)

print('> SVC results')
print(classification_report(y_test, y_pred_svc))
print(confusion_matrix(y_test, y_pred_svc))
print(f'Best params: {svc_CV.best_params_}', '\n')


# RandomForestClassifier
rf = RandomForestClassifier(random_state = 1, n_jobs=-1)
rf_CV_pipeline = make_pipeline(mct, rf)

cv = StratifiedKFold(n_splits = 3, shuffle = True, random_state = 1)

rf_param_grid = {
    'randomforestclassifier__n_estimators':[100,250,500],
    'randomforestclassifier__max_depth':[10,25,50],
    'randomforestclassifier__min_samples_split':[2,5,10],
    'randomforestclassifier__min_samples_leaf':[2,5,10],
    'randomforestclassifier__class_weight':[None, 'balanced']
}
rf_CV = GridSearchCV(rf_CV_pipeline, rf_param_grid, cv = cv, n_jobs =-1)

rf_CV.fit(X_train, y_train)
y_pred_rf = rf_CV.predict(X_test)

print('> RandomForest results')
print(classification_report(y_test, y_pred_rf))
print(confusion_matrix(y_test, y_pred_rf))
print(f'Best params: {rf_CV.best_params_}', '\n')



#DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state = 1)
dt_CV_pipeline = make_pipeline(mct, dt)

cv = StratifiedKFold(n_splits = 3, shuffle = True, random_state = 1)

dt_param_grid = {
    'decisiontreeclassifier__max_depth': [3, 5, 10, None],
    'decisiontreeclassifier__min_samples_leaf': [1, 5, 10],
    'decisiontreeclassifier__class_weight': [None, 'balanced']
}

dt_CV = GridSearchCV(dt_CV_pipeline, dt_param_grid, cv = cv, n_jobs = -1)
dt_CV.fit(X_train, y_train)

y_pred_dt = dt_CV.predict(X_test)

print('> DecisionTree results')
print(classification_report(y_test, y_pred_dt))
print(confusion_matrix(y_test, y_pred_dt))
print(f'Best params: {dt_CV.best_params_}', '\n')


#KNeighborsClassifier
kn = KNeighborsClassifier()
kn_CV_pipeline = make_pipeline(mct, kn)

cv = StratifiedKFold(n_splits = 3, shuffle = True, random_state = 1)

kn_param_grid = {
    'kneighborsclassifier__n_neighbors': [3, 5, 7],
    'kneighborsclassifier__weights': ['uniform', 'distance']
}

kn_CV = GridSearchCV(kn_CV_pipeline, kn_param_grid, cv= cv, n_jobs = -1)
kn_CV.fit(X_train, y_train)

y_pred_kn = kn_CV.predict(X_test)

print('> KNeighbors results')
print(classification_report(y_test, y_pred_kn))
print(confusion_matrix(y_test, y_pred_kn))
print(f'Best params: {kn_CV.best_params_}', '\n')
                                   


#GaussianNB
dense = FunctionTransformer(lambda X: X.toarray(), accept_sparse=True)

gnb = GaussianNB()
gnb_CV_pipeline = make_pipeline(mct,dense, gnb)

cv = StratifiedKFold(n_splits = 3, shuffle = True, random_state = 1)

gnb_param_grid = {
    'gaussiannb__var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
}

gnb_CV = GridSearchCV(gnb_CV_pipeline, gnb_param_grid, cv = cv, n_jobs= -1)
gnb_CV.fit(X_train, y_train)

y_pred_gnb = gnb_CV.predict(X_test)

print('> GaussianNB results')
print(classification_report(y_test, y_pred_gnb))
print(confusion_matrix(y_test, y_pred_gnb))
print(f'Best params: {gnb_CV.best_params_}', '\n')
