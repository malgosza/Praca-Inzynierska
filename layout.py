from tkinter import *

root=Tk()
root.geometry('680x450')
root.title("Małgorzata Niewiadomska Inżynieria Biomedyczna")
# root.configure(bg="blue")

def printName():
    print("Hello World!")

buttonSTART = Button(root, text='ROZPOCZNIJ', padx=10, pady=10, command=printName)
buttonSTART.pack(padx=180, pady=200)
buttonSTART.config(bg='green', fg='white')
buttonSTART.config(font=('helvetica', 20, 'underline italic'))

label=Label(text="Bezdotykowy intrefejs do rozpoznawania znaków z wykorzystaniem kamery RGB-D i sztucznych sieci neuronowych",font=("arial", 9,'bold')).place(x=15,y=40)

buttonPOMOC=Button(root,text='POMOC',padx=10,pady=10,bg='yellow',fg='black',command=printName).place(x=300,y=300)
# buttonPOMOC.pack(padx=500,pady=300)
# buttonPOMOC.config(bg='green', fg='white')
# buttonPOMOC.config(font=('helvetica', 7, 'underline italic'))

root.mainloop()

# from tkinter import *
#
#
# class App:
#     def __init__(self, master):
#         master.geometry("640x480")
#         fm = Frame(master)
#
#         Label(text="Bezdotykowy intrefejs do rozpoznawania znaków z wykorzystaniem kamery RGB-D i sztucznych sieci neuronowych").pack()
#         Button(text='Rozpocznij').pack(side=TOP, expand=YES)
#         # Button.option_add('font',('verdana',12,'bold'))
#         fm.pack(fill=BOTH, expand=YES)
#
#
# root = Tk()
# # root.option_add('*font', ('verdana', 12, 'bold'))
# root.title("Małgorzata Niewiadomska Inżynieria Biomedyczna")
# display = App(root)
# root.mainloop()