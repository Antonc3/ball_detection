import socket
import time
HOST = "10.220.8.28"
PORT = 8888

s = socket.socket(socket.AF_INET,socket.SOCK_STREAM) 
s.bind((HOST,PORT))
s.listen()
conn, addr = s.accept()
cntr = 0
with conn: 
    print("Connected by {}".format(conn))
    while True: 
        conn.send("hello")
        cntr+=1
        if cntr >= 10:
            break

