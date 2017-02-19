import cv2
import numpy as np

horizFrac = 0.5
bShiftW = 8
maxBorderW = 25
def removeSlide(incoming):
    """
    Tries to guess where the slide is and removes it
    Assume the top of the slide takes up at least half of the horizontal rows it occupies.
    Also assume the bottom border is at least half as thick as the top.
    We first try and find the top. If more than half of a row of the image is white, we're probably still scanning the top.
    After we're done with the top, we work from the left/right of the image and remove the first contiguous block of white on each side
    Uses lastXleft/lastXright to keep track of the side borders - large jumps imply gaps, so you don't remove anything
    """
    img = incoming.copy()

    h, w = img.shape
    img = img[6:h-6, 6:w-6] #get rid of fuzz
    h, w = img.shape

    #start from the top, which is a horizontal bar
    y = 5
    topH = h/9
    while y < h /9:
        y += 1
        frac1 = np.count_nonzero(img[y]) / len(img[y])
        frac2 = np.count_nonzero(img[y+1]) / len(img[y+1])
        frac = (frac1 + frac2) / 2
        if frac < horizFrac:
            topH = y
            break
    img[0:y+2, 0:w] = 0
    y += 2
    x = 0
    while x < w and img[y, x] == 0: x += 1
    lastXLeft = x if x is not w else 0
    while img[y, x] != 0:
        img[y, x] = 0
        x -= 1

    x = w - 1
    while x > 0 and img[y,x] == 0: x -= 1
    lastXRight = x if x is not 0 else w-1
    while img[y,x] != 0:
        img[y, x] = 0
        x -= 1

    bottomBegan = False
    while y < h - topH/2 and not bottomBegan:
        y += 1
        x = 0
        while x <= lastXLeft + bShiftW and img[y, x] == 0:
            x += 1
            if x == w: break
        if x <= lastXLeft + bShiftW:
            lastXLeft = x
            while img[y, x] != 0 and x < lastXLeft + maxBorderW:
                img[y, x] = 0
                x += 1
            if x > w/3: bottomBegan = True

        if not bottomBegan: #start deleting from the other side
            x = w - 1
            while img[y, x] == 0 and x >= lastXRight - bShiftW: x -= 1
            if x >= lastXRight - bShiftW:
                lastXRight = x
                while img[y, x] != 0 and x >= lastXRight - maxBorderW:
                    img[y,x] = 0
                    x -= 1
                if x<w/3: bottomBegan = True
    #erase the rest of the image and return
    img[y-2:h, 0:w] = 0

    if np.count_nonzero(img) == 0:
        #whoops! we messed up. This happens on B11
        #fall back to a very basic method
        img = incoming.copy()
        h, w = img.shape
        img = img[6:h - 6, 6:w - 6]
        h, w = img.shape
        img[:10, :] = 0
        img[h-20:, :] = 0
        img[:, :15] = 0
        img[:, w-15:] = 0
    return img