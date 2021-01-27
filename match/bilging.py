import cv2 as cv
import match.template_matching as tm
import time
import multiprocessing.dummy as multiprocessing
from PIL import ImageGrab
import numpy as np 
_POOL = multiprocessing.Pool(processes=8)

_W = 275
_H = 545
_RW = 45
_RH = 45

board_limits = []


def find_bilging_rectangle(img):
    template = cv.imread('bilging.png', 1)
    _Wa = 93
    _Ha = 42
    return tm.find_board_rectangle(img, template, _W, _H, _Wa, _Ha)


def get_bilging_rectangle(img):
    (x, y), (x1, y1), res = find_bilging_rectangle(img)
    return img[y:y1, x:x1], (y, y1), (x, x1)


def find_piece(img, pattern):
    return tm.find_piece(img, pattern)


def get_positions(top_lefts):
    s = set()
    for x, y in top_lefts:
        s.add((y // _RH, x // _RW))
    return list(s)


_NONE = '_'


def empty_board(n, k):
    return [[_NONE for j in range(k)] for i in range(n)]


def get_board_state(b_rec, patterns):
    def fp(x):
        return (x, get_positions(find_piece(b_rec, x)))

    s = time.time()
    board = empty_board(_H // _RH, _W // _RW)
    for p, poss in _POOL.map(fp, patterns):
        for i, j in poss:
            if board[i][j] != _NONE:
                print('warning clash:', (i, j), board[i][j], p.name)
            board[i][j] = p.name
    print('exec time:', time.time() - s)
    return '\n'.join(''.join(row) for row in board)


def track_board_state(screen_grabber, patterns):
    img_src = screen_grabber.grab()
    b_rec, (y, y1), (x, x1) = get_bilging_rectangle(img_src)
    state = None
    while True:
        new_state = get_board_state(img_src[y:y1, x:x1], patterns)
        if new_state != state:
            print('=======================================')
            print(new_state)
            state = new_state
        time.sleep(1)
        img_src = screen_grabber.grab()