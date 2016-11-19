from socket import *
from time import ctime

HOST=''
PORT=12346
BUFSIZ=1024
ADDR=(HOST,PORT)

timesocket = socket(AF_INET,SOCK_DGRAM)
timesocket.bind(ADDR)

while True:
    print 'waiting for connection...:'
    data , addr = timesocket.recvfrom(BUFSIZ)
    timesocket.sendto('[%s]%s'%(ctime(),data),addr)
    print '...received from and returned to:',addr

timesocket.close()