# ML_classification
This project applies machine learning (ML) techniques for classification using the [Health And Lifestyle Dataset](https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset/data) from Kaggle.
By analyzing body parameters and lifestyle factors, the goal is to build a predictive model for identifying the presence of chronic disease.  

## Project File Descriptions
1. synthetic_health_lifestyle_dataset.csv - The raw dataset containing chronic disease, body parameters and lifestyle informations.  
2. model_training.py – Python script containing the classification model training process.  
3. Visualization PNG files – Plots generated from exploratory data analysis (EDA) and saved as .png images.
4. model_results.csv – A DataFrame summarizing the performance of each classification model.

## Aim
The goal of this project is to predict the presence of chronic disease based on body parameters and lifestyle factors using classification models. 

## Data Description
ID: Unique identifier for each individual  
Age: Age of the individual (in years)  
Gender: Gender identity (Male, Female, Other)  
Height_cm: Height in centimeters  
Weight_kg: Weight in kilograms  
BMI: calculated as weight in kg / height in m²  
Smoker: Indicates whether the person is a smoker (Yes, No)  
Exercise_Freq: Frequency of physical exercise (None, 1-2 times/week, 3-5 times/week, Daily)  
Diet_Quality: Self-rated diet quality (Poor, Average, Good, Excellent)  
Alcohol_Consumption: Level of alcohol intake (None, Low, Moderate, High)  
Chronic_Disease: Whether the person has a chronic illness (Yes, No)  
Stress_Level: Self-reported stress level on a scale from 1 (low) to 10 (high)  
Sleep_Hours: Average hours of sleep per night  

## Analysis Workflow
1. Data exploring
   -Examine the data structure, fill in missing values, and use visualizations to observe the data distribution and correlations. Then perform preprocessing to organize the data into a usable format.
     
2. Data preprocessing
   -Based on the results of exploratory data analysis (EDA), perform transformations on the dataset’s columns.
   -Standardize numerical columns, and process categorical columns by addressing issues such as highly correlated features and categories with excessive cardinality (merging or grouping them when necessary).
   -Finally, convert categorical columns into numerical format using OneHotEncoder or OrdinalEncoder.  
   
3. Model training
   -Multiple classification models will be trained and compared to evaluate their performance.  
	Model selection strategy: Models are chosen from simple to complex, covering both linear and non-linear relationships in the data. The models used in this project include Logistic Regression, Decision Tree Classifier, Random Forest Classifier, Support Vector Classifier (SVC), K-Nearest Neighbors (KNN), Gaussian Naive Bayes, and HistGradientBoostingClassifier.
	
4. Model evaluation (first time)
   -The best-performing model will be determined by ROC-AUC, PR-AUC, and F1 score. Final judgment will be made by considering these metrics together with the intended application scenario.

5. Model training (with data augmentation)
   -If the model performance is unsatisfactory, data augmentation techniques will be applied to expand the training dataset and balance the class distribution of the target variable y.

6. Model evaluation (second time)
   -The best-performing model will be determined by ROC-AUC, PR-AUC, and F1 score. Final judgment will be made by considering these metrics together with the intended application scenario.



