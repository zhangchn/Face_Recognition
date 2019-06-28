import socket
import threading
import numpy as np
import cv2
from pickle import loads
from sys import exit
from os import unlink
from os.path import exists

shared = {
        'imgs': [None, None, None]
        }
def receive_socket():
    if exists('/tmp/img_sock'):
        unlink('/tmp/img_sock')
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.bind('/tmp/img_sock')
        sock.listen(1)
    except socket.error as msg:
        print("socket error: {}".format(msg))
        exit(-1)


    while True:
        conn, addr = sock.accept()
        buf = b''

        length = -1
        idx = -1
        state = 0
        while True:
            data = conn.recv(8192)
            if not data:
                break
            buf += data
            #print("incoming: {} bytes".format(len(data)))
            if state == 0:
                try:
                    sep_idx1 = buf.index(b"\n")
                    idx, length = [int(x.decode()) for x in buf[0:sep_idx1].split(b":")]
                except ValueError as v:
                    continue

                print("{}: {} bytes".format(idx, length))
                buf = buf[(sep_idx1 + 1):]
                state = 1
            if state == 1:
                if len(buf) < length:
                    #print("insufficient bytes")
                    continue
                img = loads(buf[0:length])
                if idx < 3:
                    shared['imgs'][idx] = img
                #cv2.imshow('img{}'.format(idx), img)
                #cv2.waitKey(1)
                buf = buf[length:]
                state = 0

        print('connection closed')
        conn.close()

def window_routine():
    while True:
        for i, img in enumerate(shared['imgs']):
            if img is not None:
                cv2.imshow('img{}'.format(i), img)
                cv2.waitKey(1)

def main():
    thread = threading.Thread(target=receive_socket)
    thread.start()
    window_routine()

if __name__ == '__main__':
    main()

