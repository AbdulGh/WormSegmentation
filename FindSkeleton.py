import numpy as np
import Tools as nt
import cv2
from Settings import settings

"""
These two functions implement the Zhang-Suen thinning algorithm
https://rosettacode.org/wiki/Zhang-Suen_thinning_algorithm
"""

def iteration(img, iter):
    toChange = []
    h,w = img.shape
    change = False
    for i in range(1,w-1):
        for j in range(1, h-1):
            if img.item(j,i) != 0:
                p9, p2, p3, p4, p5, p6, p7, p8 = neighbours = nt.getNeighbourList(i,j,img)
                A = 0
                for o in range(len(neighbours)):
                    if neighbours[o] == 0 and neighbours[(o+1)%len(neighbours)] != 0: A+=1
                B = np.count_nonzero(neighbours)
                m1 = (p2 * p4 * p6) if iter == 0 else (p2 * p4 * p8)
                m2 = (p4 * p6 * p8) if iter == 0 else (p2 * p6 * p8)
                if A == 1 and B >= 2 and B <= 6 and m1 == 0 and m2 == 0:
                    toChange.append((j,i))
                    change = True
    for coord in toChange:
        img.itemset(coord, 0)
    return change

def getSkel(binary):
    dst = np.zeros(binary.shape, dtype=np.uint8)
    dst[binary!=0] = 1
    change = True
    while change:
        change = iteration(dst, 0) or iteration(dst, 1)

        if settings.vid:
            cv2.imshow("Processing...", dst*255)
            cv2.waitKey(1)
    return dst * 255