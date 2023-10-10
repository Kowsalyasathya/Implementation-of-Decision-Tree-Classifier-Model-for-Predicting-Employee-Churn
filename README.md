# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Prepare your data Clean and format your data Split your data into training and testing sets

2.Define your model Use a sigmoid function to map inputs to outputs Initialize weights and bias terms

3.Define your cost function Use binary cross-entropy loss function Penalize the model for incorrect predictions

4.Define your learning rate Determines how quickly weights are updated during gradient descent

5.Train your model Adjust weights and bias terms using gradient descent Iterate until convergence or for a fixed number of iterations

6.Evaluate your model Test performance on testing data Use metrics such as accuracy, precision, recall, and F1 score

7.Tune hyperparameters Experiment with different learning rates and regularization techniques

8.Deploy your model Use trained model to make predictions on new data in a real-world application.
 

## Program:
```
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Kowsalya M
RegisterNumber:  212222230069
```
```
import pandas as pd
df=pd.read_csv("/content/Employee.csv")

df.info()

df.isnull().sum()

df.isnull()

df["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

df["salary"]=le.fit_transform(df["salary"])
df.head()

x=df[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y=df["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```
## Output:
### Initial data set:
![image](https://github.com/Kowsalyasathya/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118671457/e878c257-9b43-410e-b78e-e870f7cd3d09)
### Data info:
![image](https://github.com/Kowsalyasathya/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118671457/77ffc293-096b-448f-8129-afa3e7cedc04)
### Optimization of null values:
![image](https://github.com/Kowsalyasathya/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118671457/435f6fea-2070-4c3f-bec0-2d96983e9f00)
![image](https://github.com/Kowsalyasathya/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118671457/e362fcf2-9159-42cd-a2b5-3037f335494e)
### Assignment of x and y values:
![image](https://github.com/Kowsalyasathya/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118671457/f8b365c3-1ce5-49d4-88ac-134f39b22eab)
### LabelEncoder:
![image](https://github.com/Kowsalyasathya/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118671457/88b87a3f-2a78-41d9-800c-6cb2ed4907f6)
### Accuracy :
![image](https://github.com/Kowsalyasathya/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118671457/e9791a56-3b4d-4869-b768-36700353fdc0)
### Prediction:
![image](https://github.com/Kowsalyasathya/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118671457/bac4ee97-26d0-4314-b704-ba75181322e3)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
