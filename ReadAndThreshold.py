import cv2
import numpy as np
import glob
import HomomorphicFilter as hf
import Nodes as ep
import Tools as nt
from Settings import settings
from RemoveSlide import removeSlide

def inputImage(filename):
    """
    load image, normalize to fill dynamic range, convert to 8 bit
    """
    img = cv2.imread(filename, -1)
    cv2.normalize(img, img, 0, 65535, cv2.NORM_MINMAX)
    img = (img/256).astype(np.uint8)
    return img

def loadImage(ID):
    """
    reads both channels, deals with missing channels
    """
    first = glob.glob("Images/1649_1109_0003_Amp5-1_B_20070424_" + ID  + "_w1*")
    second = glob.glob("Images/1649_1109_0003_Amp5-1_B_20070424_" + ID + "_w2*")
    if len(first) == 0 or len(second) == 0: raise FileNotFoundError("Failed to load both channels for this ID")
    im1 = inputImage(first[0])
    im2 = inputImage(second[0])

    if settings.vid:
        cv2.imshow("Processing...", im2)
        cv2.waitKey(1)
    return [im1, im2]

def cropBin(img):
    """
    Finds the region of interest of an image and crops to it
    """
    _, thresh = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
    thresh = cv2.medianBlur(thresh, 3)
    x, y, w, h = cv2.boundingRect(thresh)
    img = img[y:y + h, x:x + w]
    return img, [x, y, w, h]

def cleanBin(binary, to, minPoints = 250):
    """
    Fills in gaps in the image
    """
    h, w = binary.shape
    visited = [[False for i in range(w)] for j in range(h)]
    for j in range(h):
        for i in range(w):
            if not visited[j][i] and binary.item(j,i) != to:
                visited[j][i] = True
                points = [(j,i)]
                fringe = [(j,i)]
                while len(fringe) != 0:
                    new = []
                    for y,x in fringe:
                        for point in nt.getNeighbourCoords(x,y,w,h):
                            ny, nx = point
                            if binary.item(point) != to and not visited[ny][nx]:
                                visited[ny][nx] = True
                                points.append((ny,nx))
                                new.append((point))
                    fringe = new
                if len(points) < minPoints:
                    for point in points: binary.itemset(point, to)
    return binary

def threshold(pair):
    im1, im2 = pair

    print("- Removing uneven illumination...")
    filtered = hf.getReflectance(im2.copy())

    if settings.vid:
        cv2.imshow("Processing...", filtered)
        cv2.waitKey(10)

    filtered = cv2.equalizeHist(filtered)

    if settings.vid:
        cv2.imshow("Processing...", filtered)
        cv2.waitKey(10)

    _, thresh1 = cv2.threshold(filtered, 30, 255, cv2.THRESH_BINARY)
    _, thresh2 = cv2.threshold(filtered, 30, 255, cv2.THRESH_BINARY_INV)

    if np.count_nonzero(thresh1) > np.count_nonzero(thresh2): thresh = thresh2
    else: thresh = thresh1

    # processing the stained worms
    stained = np.clip(im1, 0, 50)
    cv2.normalize(stained, stained, 0, 255, cv2.NORM_MINMAX)

    stained = cv2.GaussianBlur(stained, (3, 3), 0)
    _, stained = cv2.threshold(stained, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


    #sometimes the thresholded stained image is just a lot of static
    if np.count_nonzero(stained) > 0.4 * stained.size: binary = thresh
    else: binary = np.bitwise_or(thresh, stained)

    if settings.vid:
        cv2.imshow("Processing...", binary)
        cv2.waitKey(10)

    print("- Removing static...")
    binary = cleanBin(binary, 0)
    binary, crop = cropBin(binary)

    if settings.vid:
        cv2.imshow("Processing...", binary)
        cv2.waitKey(10)

    print("- Filling in holes...")
    binary = cleanBin(binary, 255, 70)
    binary = cv2.erode(binary, np.ones((3,3)))

    if settings.vid:
        cv2.imshow("Processing...", binary)
        cv2.waitKey(10)

    print("- Removing slide...")
    binary = removeSlide(binary)

    if settings.vid:
        cv2.imshow("Processing...", binary)
        cv2.waitKey(10)

    binary = cleanBin(binary, 0)
    crop[0] += 6 #removeSlide chops 6 off each end
    crop[1] += 6
    crop[2] -= 12
    crop[3] -= 12
    
    h,w = binary.shape
    #so we don't need to worry about border cases
    binary[0, :] = 0
    binary[h - 1, :] = 0
    binary[:, 0] = 0
    binary[:, w - 1] = 0

    return binary, crop

def fillSinglePixels(binary):
    gaps = []
    h, w = binary.shape
    for y in range(1, h-1):
        for x in range(1, w-1):
            if binary[y, x] == 0:
                list = nt.getNeighbourList(x, y, binary)
                nonzero = np.nonzero(list)[0]
                if len(nonzero) == 2 and min(nonzero[1]-nonzero[0], nonzero[0] + (len(list) - nonzero[1])) > 2:
                    gaps.append((x,y))
    for x,y in gaps:
        binary[y, x] = 255
    return binary
