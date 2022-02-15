import socket
import json
HOST = "10.220.8.28"
PORT = 8888

s = socket.socket(socket.AF_INET,socket.SOCK_STREAM) 
s.connect((HOST,PORT))
while True:
    msg = s.recv(1024)
    if not msg:
        break
    print("recieved: ", msg.decode())

s.close()