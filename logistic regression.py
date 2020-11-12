import pandas as pd 
import numpy  as np
import matplotlib.pyplot as plt
import seaborn as sb

# Importing required functions from sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import preprocessing
from sklearn.metrics import classification_report

# Importing the data
bnk= pd.read_csv("bank_data.csv")
bnk.head()
bnk.shape
bnk.columns
bnk.describe()

# summarize the number of unique values in each column
print(bnk.nunique())
# Categorizing the dependent variable "y" values into 0s and 1s
bnk.y= pd.factorize(bnk.y)[0]
bnk.y

bnk.y.value_counts()

# Visualizations
bnk.groupby("y").mean()
bnk.groupby("job").mean()
bnk.groupby("marital").mean()
bnk.groupby("education").mean()

# Countplots of different variables
sb.countplot(x="y", data=bnk, palette="hls")
sb.countplot(x="job", data=bnk, palette="hls")
sb.countplot(x="marital", data=bnk, palette="hls")
sb.countplot(x="education", data=bnk, palette="hls")
sb.countplot(x="default", data=bnk, palette="hls")

# Cross-tabulation of independent varibales with respect to dependent variable
pd.crosstab(bnk.job, bnk.y).plot(kind="bar")
pd.crosstab(bnk.default, bnk.y).plot(kind="bar")
pd.crosstab(bnk.education, bnk.y).plot(kind="bar")
pd.crosstab(bnk.loan, bnk.y).plot(kind="bar")
pd.crosstab(bnk.month, bnk.y).plot(kind="bar")
pd.crosstab(bnk.poutcome, bnk.y).plot(kind="bar")

# Histograms of "Age"
bnk.age.hist()
plt.title('Histogram of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.savefig('hist_age')

# Boxplots of one variable with respect to other
sb.boxplot("y", "age", data=bnk, palette="hls")
sb.boxplot("job", "age", data=bnk, palette="hls")
sb.boxplot("education", "age", data=bnk, palette="hls")
sb.boxplot("loan", "age", data=bnk, palette="hls")

# Removing the unwanted columns from the dataset
bnk.columns
# The columns which are less useful and carries less information are removed like contact, month, day_of_week, duration, campaign, pdays, previous, emp.var.rate,cons.price.idx,cons.conf.idx, euribor3m, nr.employed
bank= bnk.iloc[:,[0,1,2,3,4,5,6,14,20]]

# Creating the dummy variables for categories
bank.job=pd.factorize(bnk.job)[0]
bank.marital=pd.factorize(bnk.job)[0]
bank.education=pd.factorize(bnk.job)[0]
bank.default=pd.factorize(bnk.job)[0]
bank.housing=pd.factorize(bnk.job)[0]
bank.loan=pd.factorize(bnk.job)[0]
bank.poutcome=pd.factorize(bnk.job)[0]

# Dataframe after creating the dummy variables
bank.head()

# To get the count of null values in the data
bank.isnull().sum() 

# Splitting the data based on independent and dependent variables
X = bank.loc[:, bank.columns != 'y']
Y = bank.loc[:, bank.columns == 'y']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
columns = X_train.columns

# Building the LogisticRegression  model
classifier= LogisticRegression()
classifier.fit(X_train, Y_train)
classifier.coef_

classifier.predict_proba(X_train)               # Probability values

y_pred = classifier.predict(X_test)
y_pred
y_pred.shape

# Confusion matrix
from sklearn.metrics import confusion_matrix
conf_mat = confusion_matrix(Y_test["y"],y_pred)       
conf_mat
pd.crosstab(y_pred,Y_test["y"])

# Accuracy of the model
accuracy = sum(Y_test["y"]==y_pred)/Y_test.shape[0]
accuracy
