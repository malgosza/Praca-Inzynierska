from tkinter import *

def displayresult(result):
    root=Tk()
    root.geometry('680x450')
    root.title("Malgorzata Niewiadomska Inzynieria Biomedyczna")

    Label(text="ROZPOZNANO: \n",
                font=("arial", 20,'bold')).place(x=250,y=60)

    Label(text=result, fg="green",
                font=("arial", 120,'bold')
                ).place(x=300,y=150)

    root.resizable(0, 0)
    root.mainloop()

