from socket import *
from time import ctime

HOST=''
PORT=12345
BUFSIZ=1024
ADDR=(HOST,PORT)

sockettime = socket(AF_INET,SOCK_STREAM)
sockettime.bind(ADDR)
sockettime.listen(5)
while True :
    print'waiting for connection ...'
    time1,addr = sockettime.accept()
    print'connecting for...:',addr

    while True:
        data = time1.recv(BUFSIZ)
        print data
        if not data:
            break
        time1.send('[%s]%s'%(ctime(),data))
    time1.close()
sockettime.close()

