## Diabetes Prediction Application using Deep Learning  


### Table of Content
  * [Overview](#overview)
  * [Motivation](#motivation)
  * [Learning Objective](#Learning-Objective)
  * [Data Visualizations](#Data-Visualizations)
  * [CorrelationMatrix](#CorrelationMatrix)
  * [Model Evaluation](#Model-Evaluation)
  


### Overview 
In this project, the objective is to predict whether the person has Diabetes or not based on various features suach as 
- Number of times pregenant
- Glucose 
- Blood Presure
- Tricep skin
- Insulin
- BMI
- Diabetes pedigree function
- Age
The data set that has used in this project has taken from the [kaggle](https://www.kaggle.com/) . "This dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. The objective of the dataset is to diagnostically predict whether or not a patient has diabetes, based on certain diagnostic measurements included in the dataset.

### Motivation
The motivation was to experiment with machine learning project and offcourse this " Diabetes is an increasingly growing health issue due to our inactive lifestyle. If it is detected in time then through proper medical treatment, adverse effects can be prevented. To help in early detection, technology can be used very reliably and efficiently. Using machine learning we have built a predictive model that can predict whether the patient is diabetes positive or not.". This is also sort of fun to work on a project like this which could be beneficial for the society.


### Learning Objective
The following points were the objective of the project . If you are looking for all the following points in this repo then i have not covered all in this repo. I'm working on blog about this mini project and I'll update the link of blog about all the points in details later . (The main intention was to create an end-to-end ML project.)  
- Data gathering 
- Descriptive Analysis 
- Data Visualizations 
- Data Preprocessing 
- Data Modelling 
- Model Evaluation 
- Model Deployment


### Data Visualizations
- Histogram
![alt text](<https://github.com/minhAI2045/Predicting-diabetes1/blob/main/Data%20Visualization/Histogram.png>)
- Density plot
![alt text](<https://github.com/minhAI2045/Predicting-diabetes1/blob/main/Data%20Visualization/Density_pilot.png>)
- Looking at the graphs, we see that "Glucose", "Blood Pressure", "BMI" have a normal distribution; "Tricep skin" exponential distribution


### CorrelationMatrix
<img target="_blank" src="https://github.com/minhAI2045/Predicting-diabetes1/blob/main/Correlation%20Matrix/Correlation_Matrix.png" width=570>







###  Model Evaluation 
- RandomForestClassifier

<img target="_blank" src="https://github.com/minhAI2045/Predicting-diabetes1/blob/main/Model%20Evaluation/RandomForestClassifier.png" width=370>

                  precision    recall  f1-score   support

           0       0.79      0.79      0.79        99
           1       0.62      0.62      0.62        55
    accuracy                           0.73       154
    macro avg      0.70      0.70      0.70       154
    weighted avg   0.73      0.73      0.73       154


- Support Vector Machine

<img target="_blank" src="https://github.com/minhAI2045/Predicting-diabetes1/blob/main/Model%20Evaluation/SVC.png" width=370>



              precision    recall  f1-score   support

           0       0.77      0.82      0.79        99
           1       0.63      0.56      0.60        55

    accuracy                           0.73       154
    macro avg      0.70      0.69      0.70       154
    weighted avg   0.72      0.73      0.72       154


- KNeighborsClassifier 

<img target="_blank" src="https://github.com/minhAI2045/Predicting-diabetes1/blob/main/Model%20Evaluation/KNeighborsClassifier.png" width=370>

              precision    recall  f1-score   support

           0       0.76      0.80      0.78        99
           1       0.60      0.55      0.57        55

    accuracy                           0.71       154
    macro avg      0.68      0.67      0.67       154
    weighted avg   0.70      0.71      0.70       154















