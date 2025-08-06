import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.preprocessing import StandardScaler, RobustScaler
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

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score



#The file was downloaded from Kaggle: "Sleep Health and Lifestyle Dataset".
#https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset/data
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

#The target variable 'Chronic_Disease' has an imbalanced class distribution
#use 'stratify' to split data
#use  class_weight='balanced' whem training models
#input()

# Drop the unnecessary column "ID"
df = data.drop(columns = ['ID'])

# Check Numerical data distribution by = 'Chronic_Disease'
# Age, Height_cm, Weight_kg, BMI, Stress_Level, Sleep_Hours
fig1, axes1 = plt.subplots(nrows=2, ncols=3, figsize=(14,7))

#Age boxplot
sns.boxplot(df, x='Chronic_Disease', y='Age', ax = axes1[0,0])
axes1[0,0].set_title('Age distribution by Chronic_Disease')
axes1[0,0].set_xlabel('Chronic_Disease')
axes1[0,0].set_ylabel('Age')


# Height_cm boxplot
sns.boxplot(df, x='Chronic_Disease', y='Height_cm', ax = axes1[0,1])
axes1[0,1].set_title('Height_cm distribution by Chronic_Disease')
axes1[0,1].set_xlabel('Chronic_Disease')
axes1[0,1].set_ylabel('Height_cm')

#Weight_kg boxplot
sns.boxplot(df, x='Chronic_Disease', y='Weight_kg', ax = axes1[0,2])
axes1[0,2].set_title('Weight_kg distribution by Chronic_Disease')
axes1[0,2].set_xlabel('Chronic_Disease')
axes1[0,2].set_ylabel('Weight_kg')

#BMI boxplot
sns.boxplot(df, x='Chronic_Disease', y='BMI', ax = axes1[1,0])
axes1[1,0].set_title('BMI distribution by Chronic_Disease')
axes1[1,0].set_xlabel('Chronic_Disease')
axes1[1,0].set_ylabel('BMI')

#Stress level
sns.boxplot(df, x='Chronic_Disease', y='Stress_Level', ax = axes1[1,1])
axes1[1,1].set_title('Stress_Level by Chronic_Disease')
axes1[1,1].set_xlabel('Chronic_Disease')
axes1[1,1].set_ylabel('Stress_Level')

#Sleep_Hours
sns.boxplot(df, x='Chronic_Disease', y='Sleep_Hours', ax = axes1[1,2])
axes1[1,2].set_title('Sleep_Hours distribution by Chronic_Disease')
axes1[1,2].set_xlabel('Chronic_Disease')
axes1[1,2].set_ylabel('Sleep_Hours')

plt.tight_layout()
plt.savefig('Numerical_data_boxplot.png', dpi=300)
#plt.show()
plt.close()

# Check Categorical data distribution by = 'Chronic_Disease'
# Gender, Smoker, Exercise_Freq, Diet_Quality, Alcohol_Consumption
fig2, axes2 = plt.subplots(nrows=2, ncols=3, figsize=(14,8))

df_Gender = df.groupby(['Chronic_Disease','Gender']).size().reset_index(name = 'count')
sns.barplot(df_Gender, x='Chronic_Disease', y='count', hue='Gender', ax = axes2[0,0])
axes2[0,0].set_title('Count of Gender by Chronic Disease Status')
axes2[0,0].set_xlabel('Chronic Disease')
axes2[0,0].set_ylabel('Count')

df_Smoker = df.groupby(['Chronic_Disease','Smoker']).size().reset_index(name = 'count')
sns.barplot(df_Smoker, x='Chronic_Disease', y='count', hue='Smoker', ax = axes2[0,1])
axes2[0,1].set_title('Count of Smoker by Chronic Disease Status')
axes2[0,1].set_xlabel('Chronic Disease')
axes2[0,1].set_ylabel('Count')

df_Exercise_Freq = df.groupby(['Chronic_Disease','Exercise_Freq']).size().reset_index(name = 'count')
sns.barplot(df_Exercise_Freq, x='Chronic_Disease', y='count', hue='Exercise_Freq', ax = axes2[0,2])
axes2[0,2].set_title('Count of Exercise_Freq by Chronic Disease Status')
axes2[0,2].set_xlabel('Chronic Disease')
axes2[0,2].set_ylabel('Count')

df_Diet_Quality = df.groupby(['Chronic_Disease','Diet_Quality']).size().reset_index(name = 'count')
sns.barplot(df_Diet_Quality , x='Chronic_Disease', y='count', hue='Diet_Quality', ax = axes2[1,0])
axes2[1,0].set_title('Count of Diet_Quality by Chronic Disease Status')
axes2[1,0].set_xlabel('Chronic Disease')
axes2[1,0].set_ylabel('Count')

df_Alcohol_Consumption = df.groupby(['Chronic_Disease','Alcohol_Consumption']).size().reset_index(name = 'count')
sns.barplot(df_Alcohol_Consumption , x='Chronic_Disease', y='count', hue='Alcohol_Consumption', ax = axes2[1,1])
axes2[1,1].set_title('Count of Alcohol_Consumption by Chronic Disease Status')
axes2[1,1].set_xlabel('Chronic Disease')
axes2[1,1].set_ylabel('Count')


plt.tight_layout()
plt.savefig('Categorical_data_barplot.png', dpi=300)
#plt.show()
plt.close()


#Correlation Heatmap
cor = df.corr(numeric_only=True)
print(cor)
print(cor.isna().sum())
plt.figure(figsize=(14,10))

sns.heatmap(
    data=cor,
    annot=True,
    fmt=".2f",
    linewidths=0.5,
    annot_kws={"size": 14, "color": "black"},
    cmap='coolwarm',
    vmin=-1, vmax=1,
    center=0
)
plt.title("Correlation Heatmap",size=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig('Correlation Heatmap.png', dpi=300)
plt.show()

#input()



############################
#Step2. data transformation
#############################

# Based on the boxplot of the numerical data, different standization methods must be used
# Columns with outliers: Height_cm, Weight_kg, BMI, Sleep_Hours --> RobustScaler
# Columns without outliers: Age, Stress_Level --> StandardScaler

# Based on the barplot of the Categorical data
# Only 'Smoker' data show some relationship with Chronic_Disease
# But still can put all the columns in model training
# Gender--> OneHotEncoder
# Smoker, Exercise_Freq, Diet_Quality, Alcohol_Consumption --> OrdinalEncoder

#data split
X = df.drop(columns = ['Chronic_Disease'])
y = df['Chronic_Disease'].map({'No':0 ,'Yes':1})
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size = 0.2, random_state = 1)

#columns for data transformation
RobustScaler_columns = ['Height_cm','Weight_kg','BMI','Sleep_Hours']
StandardScaler_columns = ['Age','Stress_Level']
one_hot_columns = ['Gender']

Smoker_rank = ['No','Yes']
Exercise_Freq_rank = ['no','1-2 times/week','3-5 times/week','Daily']
Diet_Quality_rank = ['Poor','Average','Good','Excellent']
Alcohol_Consumption_rank = ['no','Low','Moderate','High']

# mct = make_column_transformer(
#     (RobustScaler(), RobustScaler_columns),
#     (StandardScaler(), StandardScaler_columns),
#     (OneHotEncoder(handle_unknown = 'ignore'), one_hot_columns),
#     (OrdinalEncoder(categories = [Smoker_rank]), ['Smoker']),
#     (OrdinalEncoder(categories = [Exercise_Freq_rank]), ['Exercise_Freq']),
#     (OrdinalEncoder(categories = [Diet_Quality_rank]), ['Diet_Quality']),
#     (OrdinalEncoder(categories = [Alcohol_Consumption_rank]), ['Alcohol_Consumption']),
#     remainder = 'drop',
#     n_jobs= -1
# )

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
#############################
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

#LogisticRegression
lr = LogisticRegression(class_weight='balanced')
lr_param_grid = {
    'class_weight': [None, 'balanced'],
    'penalty':['l2'],
    'C':[0.01, 0.1, 1, 10],
    'max_iter':[100, 500, 1000]
}

lr_CV = GridSearchCV(lr, lr_param_grid, cv=3, n_jobs= -1)

lr_CV_pipeline = make_pipeline(mct, lr_CV)

lr_CV_pipeline.fit(X_train, y_train)
y_pred_lr = lr_CV_pipeline.predict(X_test)

print('> LogisticRegression results')
print(classification_report(y_test,y_pred_lr))
print(confusion_matrix(y_test,y_pred_lr))
print(lr_CV.best_params_,'\n')


#SVC
svc = SVC(class_weight='balanced')
svc_param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly']
}
svc_CV = GridSearchCV(svc, svc_param_grid, cv =3, n_jobs=-1)
svc_CV_piprline =make_pipeline(mct, svc_CV)

svc_CV_piprline.fit(X_train, y_train)
y_pred_svc = svc_CV_piprline.predict(X_test)

print('> SVC results')
print(classification_report(y_test,y_pred_svc))
print(confusion_matrix(y_test,y_pred_svc))
print(svc_CV.best_params_, '\n')


# RandomForest
rf = RandomForestClassifier(class_weight='balanced')
rf_param_grid = {
    'n_estimators':[100,250,500],
    'max_depth':[10,25,50],
    'min_samples_split':[2,5,10],
    'min_samples_leaf':[2,5,10],
    'class_weight':[None, 'balanced']
}
rf_CV = GridSearchCV(rf,rf_param_grid, cv =3, n_jobs =-1)
rf_CV_pipeline = make_pipeline(mct, rf_CV)

rf_CV_pipeline.fit(X_train, y_train)
y_pred_rf = rf_CV_pipeline.predict(X_test)

print('> RandomForest results')
print(classification_report(y_test,y_pred_rf))
print(confusion_matrix(y_test,y_pred_rf))
print(rf_CV.best_params_,'\n')



#DecisionTreeClassifier
dt = DecisionTreeClassifier(class_weight='balanced')
dt_param_grid = {
    'max_depth':[3, 5, 10, None],
    'min_samples_leaf':[1, 5, 10],
    'class_weight':[None, 'balanced'],
    'random_state':[1]
}

dt_CV = GridSearchCV(dt, dt_param_grid, cv=3, n_jobs = -1)
dt_CV_pipeline = make_pipeline(mct, dt_CV)
dt_CV_pipeline.fit(X_train, y_train)

y_pred_df = dt_CV_pipeline.predict(X_test)

print('> DecisionTree results')
print(classification_report(y_test,y_pred_df))
print(confusion_matrix(y_test,y_pred_df))
print(dt_CV.best_params_,'\n')


#KNeighborsClassifier





# #GaussianNB
# gnb = GaussianNB()
# gnb_param_grid = {
#     'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
# }
# gnb_CV = GridSearchCV(gnb, gnb_param_grid, cv=3, n_jobs= -1)
# gnb_CV_pipeline = make_pipeline(mct, gnb_CV)
# gnb_CV_pipeline.fit(X_train, y_train)

# y_pred_gnb = gnb_CV_pipeline.predict(X_test)
# print(classification_report(y_test, y_pred_gnb))
# print(confusion_matrix(y_test, y_pred_gnb))
# print(gnb_CV.best_params_)



