from tkinter import *

root=Tk()
root.geometry('680x450')
root.title("Małgorzata Niewiadomska Inżynieria Biomedyczna")

def printName():
    print("Hello World!")

buttonSTART = Button(root, text='ROZPOCZNIJ', padx=10, pady=10, command=printName)
buttonSTART.pack(padx=180, pady=200)
buttonSTART.config(bg='gray', fg='white')
buttonSTART.config(font=('helvetica', 25))

label=Label(text="Bezdotykowy intrefejs do rozpoznawania znaków \nz wykorzystaniem kamery RGB-D i sztucznych sieci neuronowych",
            font=("arial", 15,'bold')).place(x=15,y=40)

buttonPOMOC=Button(root,text='POMOC',padx=10,pady=10,bg='yellow',fg='black',command=printName).place(x=300,y=300)

root.mainloop()