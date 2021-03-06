from PIL import ImageGrab, Image
import cv2 as cv
import numpy as np
import match.template_matching as tm
import match.bilging as b
from mss import mss


def grab_screen():
    img_src = ImageGrab.grab()
    return cv.cvtColor(np.array(img_src.convert('RGB')), cv.COLOR_RGB2BGR)


class ScreenGrabber(object):
    def grab(self):
        return grab_screen()


class ScreenshotGrabber(object):
    def grab(self):
        return cv.imread('screenshot.png')


class FastScreenGrabber(object):
    def grab(self):
        with mss() as sct:
            monitor = sct.monitors[1]
            sct_img = sct.grab(monitor)
            # Convert to PIL/Pillow Image
            return cv.cvtColor(np.array(Image.frombytes('RGB', sct_img.size, sct_img.bgra, 'raw', 'BGRX').convert('RGB')), cv.COLOR_RGB2BGR)


paths = [("A", './images/whiteblue_square.png', True, 0.9),
         ("B", './images/greenblue_diamond.png', True, 0.9),
         ("C", './images/lightblue_circle.png', True, 0.9),
         ("D", './images/lightyellow_circle.png', True, 0.9),
         ("E", './images/darkblue_square.png', True, 0.9),
         ("F", './images/lightblue_square.png', True, 0.9),
         ("G", './images/lightblue_diamond.png', True, 0.9),
         ("X", './images/puffer.png', False, 0.5),
         ("Y", './images/crab.png', False, 0.5),
         ("Z", './images/jellyfish.png', False, 0.5)]

patterns = [tm.build_pattern(p, n, shape=(45, 45), circle_mask=c, threshold=t) 
            for n, p, c, t in paths]

b.track_board_state(ScreenshotGrabber(), patterns)
