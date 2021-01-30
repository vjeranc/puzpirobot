from collections import namedtuple
import numpy as np
import itertools as it
from random import shuffle
# from numba import jit

# the closest neighbors are visited first
_HMOVE = (0, 1)
_VMOVE = (1, 0)

SearchState = namedtuple('SearchState', ['h', 'v'])


def in_range(x, n):
    return 0 <= x and x < n


def convert(board):
    board = np.array(board)
    print(board)
    els = unique_els(board)
    n, q = board.shape
    mapping = dict((el, i) for i, el in enumerate(els.tolist()))
    print(mapping)
    num_board = np.zeros((n, q), dtype='u1')
    for i, j in it.product(range(n), range(q)):
        num_board[i, j] = mapping[board[i, j]]
    return num_board


def get_search_state(num_board):
    return SearchState(h=neighbor_counts(num_board, _HMOVE),
                       v=neighbor_counts(num_board, _VMOVE))


def update_counts_after_swap(num_board, h_cnts, v_cnts, i, j):
    # assuming i, j is the left corner of horizontal swap
    # i, j & i, j+1 got swapped
    # horizontal move updates:
    # i, j | i, j +- m | i, j + 3
    # vertical move updates:
    # i +- m, j | i +- m, j + 1
    # where m is 0, 1 or 2 and the resulting field is in range
    update_counts(num_board, h_cnts, i, j + 3, _HMOVE)
    for dx in (1, 2, -1, -2):
        update_counts(num_board, h_cnts, i, j + dx, _HMOVE)
        update_counts(num_board, v_cnts, i + dx, j, _VMOVE)
        update_counts(num_board, v_cnts, i + dx, j + 1, _VMOVE)


def update_counts(num_board, cnts, i, j, move):
    n, q = num_board.shape
    if not in_range(i, n) or not in_range(j, q):
        return
    cnts[i, j, :] = 0
    x, y = move  # one is 1, other is 0
    for m in (1, 2, -1, -2):
        am = abs(m)
        x1 = x * m
        y1 = y * m
        if not in_range(i + x1, n) or not in_range(j + y1, q):
            continue
        el = num_board[i + x1, j + y1]
        if abs(m) != 2:
            cnts[i, j, el] += 1
            continue
        # if preceding neighbor did not have this element then skip
        if num_board[i + x1 // am, j + y1 // am] != el:
            continue
        cnts[i, j, el] += 1


def neighbor_counts(num_board, move=None):
    move = move or _HMOVE
    els = unique_els(num_board)
    el_cnt = len(els)
    n, q = num_board.shape
    cnts = np.zeros((n, q, el_cnt), dtype='u1')
    for i, j in it.product(range(n), range(q)):
        update_counts(num_board, cnts, i, j, move)
    return cnts


def unique_els(board):
    return np.unique(board)


def clear_count(num_board, search_state, x, y):
    h_cnts, v_cnts = search_state.h, search_state.v
    el1, el2 = num_board[x, y], num_board[x, y + 1]
    cnt = 0
    cnt += h_cnts[x, y, el2] if h_cnts[x, y, el2] > 2 else 0
    cnt += h_cnts[x, y + 1, el1] if h_cnts[x, y + 1, el1] > 2 else 0
    cnt += (v_cnts[x, y, el2] + 1) if v_cnts[x, y, el2] > 1 else 0
    cnt += (v_cnts[x, y + 1, el1] + 1) if v_cnts[x, y + 1, el1] > 1 else 0
    return cnt


def do_swap(num_board, search_state, i, j):
    num_board[i, j], num_board[i, j + 1] = num_board[i, j + 1], num_board[i, j]
    update_counts_after_swap(num_board, search_state.h, search_state.v, i, j)


def undo_swap(num_board, search_state, i, j):
    do_swap(num_board, search_state, i, j)


def find_best_move(board, depth=3):
    num_board = convert(board)
    n, q = num_board.shape
    ss = get_search_state(num_board)
    indices = list(it.product(range(n), range(q - 1)))
    shuffle(indices)

    def dfs(i, j, depth):
        cnt = clear_count(num_board, ss, i, j)
        if cnt > 0 or depth == 1:
            return cnt, [(i,  j)]
        do_swap(num_board, ss, i, j)
        max_moves, max_cnt = [], 0
        for x, y in indices:
            if x == i and y == j:
                continue
            if num_board[x, y] == num_board[x, y + 1]:
                continue
            cnt, moves = dfs(x, y, depth - 1)
            if cnt > max_cnt:
                max_moves = [(i, j)] + moves
                max_cnt = cnt
        undo_swap(num_board, ss, i, j)
        return max_cnt, max_moves

    sols = []
    for x, y in indices:
        cnt, moves = dfs(x, y, 3)
        sols.append((-cnt, len(moves), moves))

    return sorted(sols)[:3]
