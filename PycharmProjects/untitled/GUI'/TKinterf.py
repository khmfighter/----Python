from Tkinter import *

top = Tk()
label = Label(top,text='Hello World!!!')
label.pack()
quit = Button(top,text="HelloWorld",command = top.quit,bg = 'red',fg = 'white')
quit.pack(fill=X,expand = 1)
top.mainloop()