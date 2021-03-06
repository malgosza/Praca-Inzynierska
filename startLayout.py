from tkinter import *
from tkinter import messagebox
from opennidemo import startApp

root=Tk()
root.geometry('680x450')
root.title("Małgorzata Niewiadomska Inżynieria Biomedyczna")

def startApplication():
    root.destroy()
    startApp()

def printPOMOC():
    messagebox.showinfo("INSTRUKCJA OBSŁUGI PROGRAMU", "1. Sprawdź czy kamera jest włączona.\n"
                                      "2. Stań w odległośći około 2 m od kamery.\n"
                                      "3. Nakreśl znak.\n"
                                      "4. Poczekaj na wynik.")

buttonSTART = Button(root, text='ROZPOCZNIJ', padx=10, pady=10, command=startApplication)
buttonSTART.pack(padx=180, pady=200)
buttonSTART.config(bg='gray', fg='white')
buttonSTART.config(font=('helvetica', 25))

label=Label(text="Bezdotykowy intrefejs do rozpoznawania znaków \nz wykorzystaniem kamery RGB-D i sztucznych sieci neuronowych",
            font=("arial", 15,'bold')).place(x=15,y=40)

buttonPOMOC=Button(root,text='POMOC',padx=10,pady=10,bg='yellow',fg='black',command=printPOMOC).place(x=300,y=300)

root.resizable(0, 0)
root.mainloop()