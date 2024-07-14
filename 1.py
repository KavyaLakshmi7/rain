
#Inporting required packages
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.model_selection import train_test_split,cross_val_score
from tkinter import *
from tkinter import messagebox
from tkinter import ttk
from PIL import ImageTk,Image

df = pd.read_csv(r'C:\Users\Kavya\Desktop\project\weatherAUS.csv')
df=df.dropna()
df=df.drop(['Evaporation','Sunshine','WindGustDir','WindGustSpeed','WindDir9am',
            'WindSpeed9am','Humidity9am','Pressure9am','Pressure3pm','Cloud9am',
            'Cloud3pm','Temp9am','WindDir3pm'],axis=1)



df['Month'] = pd.to_datetime(df['Date'], errors='coerce')
df['Month']=df['Month'].dt.month

df=df.drop(['Date'],axis=1)

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df['Location']=le.fit_transform(df['Location'])
df['RainToday']=le.fit_transform(df['RainToday'])
df['RainTomorrow']=le.fit_transform(df['RainTomorrow'])

x=df.drop(['RainTomorrow'],axis=1)
y=df['RainTomorrow']
df.columns
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
sr=sc.fit_transform(x)
x=sr

xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100,random_state=10)
model.fit(xtrain,ytrain)

print(model.score(xtrain,ytrain))


from tkinter import *

import tkinter as tk
from functools import partial

def Decision():
    print("yyy")
    input=(int(Location.get()),float(Mintemp.get()),float(Maxtemp.get()),float(Humidity.get()),float(Temperature.get()),float(Rain.get()),float(Month.get()),float(Windspeed.get()),int(Raintoday.get()))
    arr=np.asarray(input)
    print("xxx")
    reshape_ar=arr.reshape(1,-1)
    sr=sc.transform(reshape_ar)
    pred=model.predict(sr)
    print(pred)
    
    

    if(pred[0]==1):
        res=Label(top,text="rain predicted")
        res.place(x=470,y=160)
    else:
        res=Label(top,text="no rain")
        res.place(x=470,y=160)
  
top = Tk()
width=top.winfo_screenwidth()
height=top.winfo_screenheight()

top.title("RAINFALL PREDICTION")
top.geometry("700x500")
bg=PhotoImage(file="bg.png")

Location=tk.StringVar()
Mintemp=tk.StringVar()
Maxtemp=tk.StringVar()
Humidity=tk.StringVar()
Temperature=tk.StringVar()
Rain=tk.StringVar()
Month=tk.StringVar()
Windspeed=tk.StringVar()
Raintoday=tk.StringVar()

label1=Label(top,image=bg)
label1.place(x=0,y=0)
res=Label(top)
res.place(x=470,y=160)
Loc= Label(top, text = "Location")
Loc.place(x =10 ,y = 20)  
MinT = Label(top, text = "MinTemp")
MinT.place(x = 10, y = 60)  
MaxT = Label(top, text = "MaxTemp")
MaxT.place(x = 10, y = 100)
Humid = Label(top, text = "Humidity")
Humid.place(x = 10, y = 140) 
Temp = Label(top, text = "Temperature")
Temp.place(x = 10, y = 190) 
Rn = Label(top, text = "Rain")
Rn.place(x = 10, y = 230)
Mon= Label(top, text = "Month")
Mon.place(x = 10, y = 270)
Wndspd = Label(top, text = "Windspeed")
Wndspd.place(x = 10, y = 310)
Rntdy = Label(top, text = "Raintoday")
Rntdy.place(x = 10, y = 350)
frame = Frame(top)
frame.pack(pady=20)
bmitbtn = Button(top, text = "Submit",activebackground = "pink",
                 activeforeground = "blue",command=Decision).place(x = 30, y = 400)
result=Label(top)


e1 = Entry(top,textvariable=Location).place(x = 100, y = 20)  
e2 = Entry(top,textvariable=Mintemp).place(x = 100, y = 60)  
e3 = Entry(top,textvariable=Maxtemp).place(x = 100, y = 100) 
e4 = Entry(top,textvariable=Humidity).place(x = 100, y = 140) 
e5 = Entry(top,textvariable=Temperature).place(x = 100, y = 190) 
e6 = Entry(top,textvariable=Rain).place(x = 100, y = 230) 
e7 = Entry(top,textvariable=Month).place(x = 100, y = 270) 
e8 = Entry(top,textvariable=Windspeed).place(x = 100, y = 310) 
e9 = Entry(top,textvariable=Raintoday).place(x = 100, y = 350)

    
    


top.mainloop()

