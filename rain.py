#IMPORTING LIBRARIES AND DATA SET
import numpy as np
import pandas as pd

print("INDEPENDENT VARIABLE\n")
print(X)
print("\n\nDEPENDENT VARIABLE")
print(Y)

#PREPROCESSING
Y = Y.reshape(-1,1)
#HANDLING MISSING VALUES
from sklearn.impute import SimpleImputer
i = SimpleImputer(missing_values=np.nan,strategy='most_frequent')
X = i.fit_transform(X)
Y = i.fit_transform(Y)
print("\n\n******preprocessing*********")
print("\n\n 1)AFTER HANDLING MISSING VALUES")
print("\n INDEPENDENT VARIABLE")
print(X)
print("\n\nDEPENDENT VARIABLE")
print(Y)


#ENCODING
from sklearn.preprocessing import LabelEncoder
l1 = LabelEncoder()
X[:,0] = l1.fit_transform(X[:,0])
l2 = LabelEncoder()
X[:,4] = l2.fit_transform(X[:,4])
l3 = LabelEncoder()
X[:,6] = l3.fit_transform(X[:,6])
l4 = LabelEncoder()
X[:,7] = l4.fit_transform(X[:,7])
l5 = LabelEncoder()
X[:,-1] = l5.fit_transform(X[:,-1])
l6 = LabelEncoder()
Y[:,-1] = l6.fit_transform(Y[:,-1])

print("\n2)AFTER ENCODING")
print("\n INDEPENDENT VARIABLE")
print(X)
print("\n\nDEPENDENT VARIABLE")
print(Y)


#SCALING
Y = np.array(Y,dtype=float)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
     
print("\n3)AFTER SCALING")
print(X)

print("\n\n************Machine learning***************\n\n")
#SPLITING
from sklearn.model_selection import train_test_split
Xtrain,Xtest,Ytrain,Ytest = train_test_split(X,Y,test_size=0.2,random_state=0)
     
'''print("TRAINING SET\n")
print(Xtrain)
print(Ytrain)

print("TESTING SET\n")
print(Xtest)
print(Ytest)'''



#TRAINING


'''from sklearn.ensemble import DecisionTreeClassifier
model= DecisionTreeClassifier(n_estimators=100,random_state=10)
model.fit(X_train,Y_train)
predictions=model.predict(X_train)

print(accuracy_score(Y_train, predictions))'''
Ytrain=Ytrain.ravel()
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100,random_state=10)
model.fit(Xtrain,Ytrain)

print(model.score(Xtrain,Ytrain))
ypred=model.predict(Xtest)

print(ypred)
from sklearn.metrics import confusion_matrix  
cm= confusion_matrix(Ytest, ypred)
print("\nCONFUSION MATRIX\n")
print(cm)

print("ytest")
print(Ytest)
print(ypred)
Ytest=Ytest.ravel()
ypred = l6.inverse_transform(np.array(model.predict(Xtest),dtype=int))
Ytest = l6.inverse_transform(np.array(Ytest,dtype=int))


print(ypred)

print(Ytest)

'''from sklearn.metrics import confusion_matrix  
cm= confusion_matrix(Ytest, ypred)
print("\nCONFUSION MATRIX\n")
print(cm)'''


ypred = ypred.reshape(-1,1)
Ytest = Ytest.reshape(-1,1)

df = np.concatenate((Ytest,ypred),axis=1)
dataframe = pd.DataFrame(df,columns=['   Rain on Tommorrow   ','  Predition of Rain'])

print("\n\n PREDICTION\n\n")
print(dataframe)

from sklearn.metrics import accuracy_score
print("\n\nACCURACY")
print(accuracy_score(Ytest,ypred))





     
