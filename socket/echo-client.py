import socket

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect(('10.220.8.28', 8888))
    s.sendall(b'Hello world')
    data = s.recv(1024)

print('Received', repr(data))