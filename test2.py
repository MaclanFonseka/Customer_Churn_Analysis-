# -*- coding: utf-8 -*-
"""
Created on Sat May 23 11:24:50 2020

@author: Maclan Samanga
"""

#importing libraries
import pandas as pd #data processing
import matplotlib.pyplot as plt
import seaborn as sns

#load the data 
mydataset = pd.read_csv (r"C:\Users\Maclan Samanga\Desktop\Practical Work_2\Telco-Customer-Churn.csv")
mydataset.head()

#get the number of rows and cols
mydataset.shape

#datatypes of column 
print(mydataset.dtypes)

#get count of empty cells in cols
print(mydataset.isna().sum())

#check for any missing null values
print (mydataset.isnull().values.any())

#view satistics of my dataset
print(mydataset.describe())

#converting TotalCharges into float type
mydataset['TotalCharges'] = pd.to_numeric(mydataset['TotalCharges'], errors='coerce')
mydataset['TotalCharges'] = mydataset['TotalCharges'].fillna(mydataset['TotalCharges'].median())

# Convert SeniorCitizen from integer to string
mydataset['SeniorCitizen'] = mydataset['SeniorCitizen'].apply(lambda x: 'Yes' if x==1 else 'No')

#count of churn
print (mydataset['Churn'].value_counts())

#visualzing the churn rate 
sns.countplot(mydataset['Churn'])


#print all values of data types and their unique values 
for column in mydataset.columns:
    if mydataset[column].dtypes==object:
        print(str(column)+ ' : '+ str(mydataset[column].unique()))
        print (mydataset[column].value_counts())
        print('------------------------------')
        

# Exploratory analysis on non-continuous features
plt.figure(figsize=(15, 18))

plt.subplot(4, 2, 1)
sns.countplot('gender', data=mydataset, hue='Churn')

plt.subplot(4, 2, 2)
sns.countplot('SeniorCitizen', data=mydataset, hue='Churn')

plt.subplot(4, 2, 3)
sns.countplot('Partner', data=mydataset, hue='Churn')

plt.subplot(4, 2, 4)
sns.countplot('Dependents', data=mydataset, hue='Churn')

plt.subplot(4, 2, 5)
sns.countplot('PhoneService', data=mydataset, hue='Churn')

plt.subplot(4, 2, 6)
sns.countplot('PaperlessBilling', data=mydataset, hue='Churn')
    
plt.subplot(4, 2, 7)
sns.countplot('StreamingMovies', data=mydataset, hue='Churn')

plt.subplot(4, 2, 8)
sns.countplot('StreamingTV', data=mydataset, hue='Churn')

plt.figure(figsize=(15, 18))

plt.subplot(4, 2, 1)
sns.countplot('InternetService', data=mydataset, hue='Churn')

plt.subplot(4, 2, 2)
sns.countplot('DeviceProtection', data=mydataset, hue='Churn')

plt.subplot(4, 2, 3)
sns.countplot('TechSupport', data=mydataset, hue='Churn')

plt.subplot(4, 2, 4)
sns.countplot('OnlineSecurity', data=mydataset, hue='Churn')

plt.subplot(4, 2, 5)
sns.countplot('OnlineBackup', data=mydataset, hue='Churn')

plt.subplot(4, 2, 6)
sns.countplot('MultipleLines', data=mydataset, hue='Churn')

plt.subplot(4, 2, 7)
g = sns.countplot('PaymentMethod', data=mydataset, hue='Churn')
g.set_xticklabels(g.get_xticklabels(), rotation=45);

plt.subplot(4, 2, 8)
g = sns.countplot('Contract', data=mydataset, hue='Churn')
g.set_xticklabels(g.get_xticklabels(), rotation=45);


"""It seems that the gender column doesn't have a big effect on the Chur rate.

Churn: 50.73% Males, 49.26% Females
Not Churn: 50.24% Males, 49.75% Females

We can drop the varibale 'gender' as it doesn't effect on churning'"""

#remove some useless columns
mydataset = mydataset.drop('customerID', axis = 1)
mydataset = mydataset.drop('gender', axis =1)

#get correlation between data
mydataset.corr()

#visualization of correlated data
plt.figure(figsize=(10,10))
sns.heatmap(mydataset.corr(), annot=True)
#Due ToatlCharges highly correlated with MonthlyChrage and tenure, remove TotalCharge
mydataset = mydataset.drop('TotalCharges', axis =1)


#transform the data 
#transform non-numerical data into numerical data
str_list = [] # empty list to contain columns with strings (words)
for colname, colvalue in mydataset.iteritems():
    if type(colvalue[1]) == str:
         str_list.append(colname)
# Get to the numeric columns by inversion            
num_list = mydataset.columns.difference(str_list) 


mydataset.head()

# Create dummy variables for features with more than two classes
dummy_data = pd.get_dummies(mydataset,drop_first=True)
dummy_data.head()

#split data 
X = dummy_data.iloc[:, 0:28].values
Y = dummy_data.iloc[:, 28].values

#split the data into 75% training and 25% testing
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.25, random_state = 0)

#preprocessing
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

#logistic regression
from sklearn.linear_model import LogisticRegression
logistic_reg = LogisticRegression(random_state = 0)
logistic_reg.fit(X_train, Y_train)
y_pred = logistic_reg.predict(X_test)
acc_lg = accuracy_score(Y_test, y_pred)
print("Accuracy: {}".format(acc_lg))
print()
print(classification_report(Y_test,y_pred))
cnf_matrix = metrics.confusion_matrix(Y_test,y_pred)
print(cnf_matrix)

#Decision Trees
from sklearn.tree import DecisionTreeClassifier
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
y_pred = decision_tree.predict(X_test)
acc_dt = accuracy_score(Y_test, y_pred)
print("Accuracy: {}".format(acc_dt))
print()
print(classification_report(Y_test,y_pred))
cnf_matrix = metrics.confusion_matrix(Y_test,y_pred)
print(cnf_matrix)


#Support Vector Machine
from sklearn.svm import SVC
svc_cl = SVC(kernel = 'rbf', random_state = 0)
svc_cl.fit(X_train, Y_train)
y_pred = svc_cl.predict(X_test)
acc_svm = accuracy_score(Y_test, y_pred)
print("Accuracy: {}".format(acc_svm))
print()
print(classification_report(Y_test,y_pred))
cnf_matrix = metrics.confusion_matrix(Y_test,y_pred)
print(cnf_matrix)


#Naive Bayes 
from sklearn.naive_bayes import GaussianNB
gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
y_pred = gaussian.predict(X_test)
acc_nb = accuracy_score(Y_test, y_pred)
print("Accuracy: {}".format(acc_nb))
print()
print(classification_report(Y_test,y_pred))
cnf_matrix = metrics.confusion_matrix(Y_test,y_pred)
print(cnf_matrix)

models_acc = pd.DataFrame({
    'Models': ['Decision Tree', 'Logistic Regression', 
               'Support Vector Machine','Naive Bayes'],
    'Accuracy': [acc_dt,acc_lg,acc_svm,acc_nb] })

print(models_acc.sort_values(by='Accuracy',ascending=False))

  
        

