from collections import namedtuple
import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
from numba import jit
import time


_MATCHING_THRESHOLD = 0.9

Pattern = namedtuple('Pattern', ['piece', 'mask', 'name', 'threshold'])


def build_pattern(file_path, name=None, shape=None, circle_mask=False, 
                  threshold=_MATCHING_THRESHOLD):
    name = name or file_path
    img = cv.imread(file_path, cv.IMREAD_UNCHANGED)
    if shape:
        img = cv.resize(img, shape)
    (_, mask) = create_mask(img)
    if circle_mask:
        # circle mask removes confusion around cursor corners
        # [el][el]
        (_, cmask) = create_circle_mask(img)
        mask = cv.bitwise_and(cmask, mask)
    return Pattern(piece=img[:, :, :3], mask=mask, name=name, 
                   threshold=threshold)


def create_circle_mask(img):
    h, w, _ = img.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    cx, cy = h//2, w//2
    r = int(h/2 * 0.5)
    return None, cv.circle(mask, (cx, cy), r, 255, thickness=cv.FILLED)


def create_mask(transparent_img, thresh=cv.THRESH_BINARY):
    return cv.threshold(transparent_img[:, :, 3], 0, 255, thresh)


@jit
def find_piece(img, pattern, coeff=cv.TM_CCOEFF_NORMED):
    # matchTempalte for full image is a bit slow because piece pattern is tried
    # outside of the regular grid boundaries
    res = cv.matchTemplate(img, pattern.piece, coeff, mask=pattern.mask)
    loc = np.where(res >= pattern.threshold)
    return [pt for pt in zip(*loc[::-1])]


def find_board_rectangle(img, head_img, width, height, width_adjustment,
                         height_adjustment):
    img = img.copy()
    # Apply template Matching
    res = cv.matchTemplate(img, head_img, cv.TM_CCOEFF)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    top_left = (max_loc[0] - width_adjustment, max_loc[1] + height_adjustment)
    bottom_right = (top_left[0] + width, top_left[1] + height)
    return (top_left, bottom_right, res)


def plot_result(img, top_left, bottom_right, res):
    cv.rectangle(img, top_left, bottom_right, 255, 2)
    plt.subplot(121), plt.imshow(res, cmap='gray')
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(img, cmap='gray')
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.suptitle('cv.TM_CCOEFF')
    plt.show()


def plot_pieces(img, piece_img):
    w, h = piece_img.shape[0], piece_img.shape[1]
    pieces = find_piece(img, piece_img)
    plot_rectangles(img.copy(), pieces, w, h)


def draw_rectangles(img, top_lefts, w, h, color=255):
    img = img.copy()
    for tl in top_lefts:
        cv.rectangle(img, tl, (tl[0] + w, tl[1] + h), color, 2)
    return img


def plot_rectangles(img, top_lefts, w, h, color=255):
    img = draw_rectangles(img, top_lefts, w, h, color)
    plt.imshow(img)
    plt.show()
