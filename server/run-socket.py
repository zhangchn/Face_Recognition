import socket
import threading
import numpy as np
import cv2
from pickle import loads
from sys import exit
from os import unlink
from os.path import exists
from time import sleep

shared = {
        'imgs': [None, None],
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
        print("connection accepted")
        buf = b''

        length = -1
        idx = -1
        state = 0
        img = None
        while True:
            data = conn.recv(81920)
            if not data:
                break
            buf += data
            if state == 0:
                try:
                    sep_idx1 = buf.index(b"\n")
                    idx_str, length_str, origin_x, origin_y, shape_str = [x.decode() for x in buf[0:sep_idx1].split(b":")]
                    idx = int(idx_str)
                    length = int(length_str)
                    origin = (int(origin_x), int(origin_y))
                    shape = [int(x) for x in shape_str.split(",")]
                    #img = np.zeros(shape=shape, dtype=np.uint8)
                except ValueError as v:
                    continue

                print("{}: {} bytes, ({}, {}), ({}, {}, {})".format(idx, length, origin[0], origin[1], shape[0], shape[1], shape[2]))
                buf = buf[(sep_idx1 + 1):]
                state = 1
            if state == 1:
                if len(buf) < length:
                    continue
                #img = loads(buf[0:length])
                #memoryview(buf[0:length])
                img = np.frombuffer(buf[0:length], dtype=np.uint8).reshape(shape)
                if idx < len(shared['imgs']):
                    shared['imgs'][idx] = (img, origin[0], origin[1], shape[0], shape[1])
                    #shared['bbox'][idx] = (origin, (origin[0] + shape[0], origin[1] + shape[1]))
                buf = buf[length:]
                state = 0

        print('connection closed')
        conn.close()

def window_routine():
    img0 = None
    img = None
    while True:
        for i, x in enumerate(shared['imgs']):
            if x is None:
                continue
            (img, origin_x, origin_y, width, height) = x
            if i == 0:
                img0 = img
                if img0 is not None:
                    img0 = img0.copy()
            if i > 0 and img0 is not None and img is not None:
                #np.copyto(img0[bbox[0][0]:bbox[1][0], bbox[0][1]:bbox[1][1]], img)
                img0[origin_y:(origin_y + width), origin_x:(origin_x + height), :] = img
        if img0 is not None:
            cv2.imshow('img', img0)
            cv2.waitKey(5)

def main():
    thread = threading.Thread(target=receive_socket)
    thread.start()
    window_routine()

if __name__ == '__main__':
    main()

