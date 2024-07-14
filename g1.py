from tkinter import *
from tkinter import messagebox
from tkinter import ttk
#from PIL import ImageTk,Image


top=Tk()
width=top.winfo_screenwidth()
height=top.winfo_screenheight()
loc=StringVar()
mint=StringVar()
maxt=StringVar()
rain=StringVar()
windsp=StringVar()
humi=StringVar()
rt=StringVar()
month=StringVar()
top.geometry(f"%dx%d"%(width,height))
top.title("RAINFALL PREDICTION")


def Decision():
    dec=Toplevel(top)
    dec.geometry('500x500')
    dec.title("Result")

input=(int(loc.get()),float(mint.get()),float(maxt.get()),float(rain.get()),
       float(windsp.get()),float(humi.get()),int(rt.get()))

