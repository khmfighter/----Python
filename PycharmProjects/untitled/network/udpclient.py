from socket import *

HOST='127.0.0.1'
PORT=12346
BUFSIZ=1024
ADDR=(HOST,PORT)

timesocket = socket(AF_INET,SOCK_DGRAM)

while True:
    data = raw_input('> ')
    if not data:
        break
    timesocket.sendto(data,ADDR)
    data,addr = timesocket.recvfrom(BUFSIZ)
    if not data:
        break
    print data
timesocket.close()