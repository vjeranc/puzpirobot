import cv2 as cv
import match.template_matching as tm
import time
import multiprocessing.dummy as multiprocessing
import optimize.search as s
import numpy as np
import matplotlib.pyplot as plt

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
    return board


def repr_board(board):
    return '\n'.join(''.join(row) for row in board)


_COLORS = [(0, 0, 255), (255, 255, 255), (255, 0, 0), (50, 50, 50)]


def draw_sol(img, moves):
    img = img.copy()
    for (idx, (i, j)) in enumerate(moves):
        cv.rectangle(img, (j * _RH, i * _RW), ((j + 1) * _RH, (i + 1) * _RW), 
                     _COLORS[idx % len(_COLORS)], thickness=cv.FILLED)
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.draw()
    plt.pause(0.001)


def track_board_state(screen_grabber, patterns):
    img_src = screen_grabber.grab()
    b_rec, (y, y1), (x, x1) = get_bilging_rectangle(img_src)
    repr_state = None
    plt.ion()
    while True:
        new_state = get_board_state(img_src[y:y1, x:x1], patterns)
        repr_new_state = repr_board(new_state)
        if repr_new_state != repr_state:
            if not np.count_nonzero(np.array(repr_new_state) == '_'):
                cnt, l, moves = s.find_best_move(new_state, depth=3)[0]
                print(moves)
                draw_sol(img_src[y:y1, x:x1], moves)
                repr_state = repr_new_state
                continue
        time.sleep(0.2)
        img_src = screen_grabber.grab() 