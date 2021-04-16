import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


#import the data
data = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

#preprocesssing, to get all the columns in the suitable format
data['Churn'] = data['Churn'].map(
                   {'Yes':1 ,'No':0})

x=data["TotalCharges"]
l=data.query("tenure ==0")
for i in range(len(x)):
    try:
     x[i]=float(x[i])
    except:
        x[i]=float(0)

data['TotalCharges']=x
data["TotalCharges"]=data["TotalCharges"].astype('float')

#checking for missing values
sns.heatmap(data.isnull(), yticklabels = False, cbar = False, cmap = 'viridis')
#no missing values


#creating a feature called close contact
y=data[["Partner","Dependents","Churn" ]]
z=[]
for i in range(len(y)):
    deg=0
    if y["Partner"][i]=="Yes":
         deg=deg+1
    if y["Dependents"][i]=="Yes":
        deg=deg+1
    z.append(deg)
y["close_contact"]=z

data["close_contact"]=y["close_contact"]

#dropping customer ID
data=data.drop(['customerID'], axis = 1)


#dividing into train-test
train=data.iloc[0:4930]
test=data.iloc[4930:]


first=train[["SeniorCitizen","tenure","InternetService","OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","MonthlyCharges","close_contact","Churn"]]
first_test=test[["SeniorCitizen","tenure","InternetService","OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","MonthlyCharges","close_contact","Churn"]]

#means of monthly charges grouped by type of the internet service
mot=first[["InternetService","MonthlyCharges"]]
avg=mot.groupby(['InternetService']).mean()


#one-hot encoding of object data type
firstnum=first._get_numeric_data()
firstob=first.select_dtypes(include=object, exclude=None)
firstob=pd.get_dummies(firstob)
first= pd.concat([firstob,firstnum], axis=1)
#plt to show corrleation between the features
sns.heatmap(first.corr())


first=first[["InternetService_DSL","InternetService_Fiber optic","OnlineSecurity_Yes","OnlineBackup_Yes","DeviceProtection_Yes", "TechSupport_Yes","SeniorCitizen","tenure","close_contact" ,"Churn"]]
X=first.drop(["Churn"], axis=1)


from statsmodels.api import add_constant
X=add_constant(X)
first_testnum=first_test._get_numeric_data()
first_testob=first_test.select_dtypes(include=object, exclude=None)
first_testob=pd.get_dummies(first_testob)
first_test= pd.concat([first_testob,first_testnum], axis=1)
first_test=first_test[["InternetService_DSL","InternetService_Fiber optic","OnlineSecurity_Yes","OnlineBackup_Yes","DeviceProtection_Yes", "TechSupport_Yes","SeniorCitizen","tenure","close_contact" ,"Churn"]]

T=first_test.drop(["Churn"], axis=1)
T=add_constant(T)

#calculating the variance inflation factor
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns


#logistic regression model 

import statsmodels.api as sm
log_reg = sm.Logit(first["Churn"],X).fit()
yhat = log_reg.predict(T)
yhat=list(yhat)




y=train["Churn"]
from statsmodels.api import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor
trainnum=train._get_numeric_data()
trainob=train.select_dtypes(include=object, exclude=None)
trainob=pd.get_dummies(trainob, drop_first=True)
train= pd.concat([trainnum,trainob], axis=1)

#calculating vif of all the features 
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(train.values, i) for i in range(train.shape[1])]
vif["features"] = train.columns

#Features dropped one by one by subsequently caluclating vif and removing the highest contributors
train=train.drop(["StreamingTV_No internet service",'StreamingMovies_No internet service','TechSupport_No internet service','DeviceProtection_No internet service', 'OnlineBackup_No internet service','OnlineSecurity_No internet service' ,'MultipleLines_No phone service',"MonthlyCharges","TotalCharges","Churn"],axis=1)


testnum=test._get_numeric_data()
testob=test.select_dtypes(include=object, exclude=None)
testob=pd.get_dummies(testob, drop_first=True)
test= pd.concat([testnum,testob], axis=1)

testy=test["Churn"]
test=test.drop(["StreamingTV_No internet service",'StreamingMovies_No internet service','TechSupport_No internet service','DeviceProtection_No internet service', 'OnlineBackup_No internet service','OnlineSecurity_No internet service' ,'MultipleLines_No phone service',"MonthlyCharges","TotalCharges","Churn"],axis=1)

import statsmodels.api as sm
log_reg = sm.Logit(y,train).fit()
yhat = log_reg.predict(test)
yhat=list(yhat)


#evaluating the model by setting different threshold of p for y=1
def model_evaluate(threshold,yhat):
    y=[]
    test=list(testy)
    TP=0
    FP=0
    FN=0
    TN=0
    for i in range(len(yhat)):
        if yhat[i]<threshold:
            y.append(0)
        else:
            y.append(1)

        if y[i]==test[i]:
            if test[i]==0:
                TN=TN+1
            else:
                TP=TP+1
        else:
            if test[i]==1:
                FN=FN+1
            else:
                FP=FP+1

    Precision=TP/(TP+FP)
    Recall=TP/(TP+FN)
    Fscore=2*Precision*Recall/(Precision+Recall)
    Accuracy=(TP+TN)/len(yhat)
    return [Precision, Recall, Fscore,Accuracy]


score=[]
t=0.3
while t<0.8:
    score.append([t]+model_evaluate(t,yhat))


    t=t+0.05

df = pd.DataFrame(score, columns = ['Threshold', "Precision", "Recall", "Fscore","Accuracy"])
df.set_index('Threshold',inplace=True)
