import cv2 as cv
import match.template_matching as tm
import multiprocessing.dummy as multiprocessing
import optimize.search as s
import numpy as np
import matplotlib.pyplot as plt
import itertools as it

_POOL = multiprocessing.Pool(processes=8)

_W = 275
_H = 545
_RW = 45
_RH = 45
_N = _H // _RH
_Q = _W // _RW

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


def get_board_state_alt(b_rec, patterns):
    # faster when get_board_state is not parallelized
    def fp(x):
        x, y = x
        x1, x2 = x * _RH, (x + 1) * _RH + 1
        y1, y2 = y * _RW, (y + 1) * _RW + 1
        img = b_rec[x1:x2, y1:y2]
        r = []
        for p in patterns:
            ps = get_positions(find_piece(img, p))
            if not ps:
                continue
            r.append((p, ps))
        return r
    indices = it.product(range(_N), range(_Q))
    board = empty_board(_N, _Q)
    for (i, j), r in zip(indices, map(fp, indices)):
        if not r:
            continue
        if len(r) > 1:
            print('warning clash', (i, j), len(r))
        board[i][j] = r[0][0].name
    return board


def get_board_state(b_rec, patterns):
    def fp(x):
        return (x, get_positions(find_piece(b_rec, x)))
    board = empty_board(_N, _Q)
    for p, poss in map(fp, patterns):
        for i, j in poss:
            if board[i][j] != _NONE:
                print('warning clash:', (i, j), board[i][j], p.name)
            board[i][j] = p.name
    return board


def get_means(b_rec):
    img = b_rec.copy()
    for x, y in it.product(range(_N), range(_Q)):
        x1, x2 = x * _RH, (x + 1) * _RH + 1
        y1, y2 = y * _RW, (y + 1) * _RW + 1
        avg_color_per_row = np.average(b_rec[x1:x2, y1:y2], axis=0)
        avg_color = np.average(avg_color_per_row, axis=0)
        img[x1:x2, y1:y2] = avg_color
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.show()


def repr_board(board):
    return '\n'.join(''.join(row) for row in board)


_COLORS = [(0, 0, 255), (255, 255, 255), (255, 0, 0), (50, 50, 50)]


def draw_sol(img, moves):
    img = img.copy()
    for (idx, (i, j)) in enumerate(moves):
        cv.rectangle(img, (j * _RH, i * _RW), ((j + 1) * _RH, (i + 1) * _RW),
                     _COLORS[idx % len(_COLORS)], thickness=cv.FILLED)
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGBA))
    plt.draw()
    plt.pause(0.001)


def track_board_state(screen_grabber, patterns):
    img_src = screen_grabber.grab()
    b_rec, (y, y1), (x, x1) = get_bilging_rectangle(img_src)
    repr_state = None
    while True:
        new_state = get_board_state(img_src[y:y1, x:x1], patterns)
        # get_board_state_alt(img_src[y:y1, x:x1], patterns)
        repr_new_state = repr_board(new_state)
        if repr_new_state != repr_state:
            if not np.count_nonzero(np.array(repr_new_state) == '_'):
                cnt, l, moves = s.find_best_move(new_state, depth=3)[0]
                draw_sol(img_src[y:y1, x:x1], moves)
                repr_state = repr_new_state
                continue
        img_src = screen_grabber.grab()
