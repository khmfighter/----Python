from socket import *

HOST='127.0.0.1'
PORT=12345
BUFSIZ=1024
ADDR=(HOST,PORT)

sockettime = socket(AF_INET,SOCK_STREAM)
sockettime.connect(ADDR)

while True:
    data = raw_input('> ')
    if not data :
        break
    sockettime.send(data)

    data = sockettime.recv(BUFSIZ)
    if not data:
        break
    print data
sockettime.close()
