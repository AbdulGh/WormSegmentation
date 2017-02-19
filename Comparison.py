import os
import numpy as np
import cv2
import glob
import re

def cropImage(image, crop):
    x, y, w, h = crop
    return image[y:y+h, x:x+w]

def createComparisonImage(binary, truth):
    excess = binary - truth
    both = cv2.bitwise_and(binary, truth)
    undetected = truth - binary
    comparison = cv2.merge((excess, both, undetected))

    div = np.count_nonzero(truth)

    if div == 0:
        percentSuccessful = percentUndecected = percentOver = 0
    else:
        percentSuccessful = np.count_nonzero(both) / div * 100
        percentOver = ((binary != 0) & (truth == 0)).sum() / div * 100
        percentUndecected = ((binary == 0) & (truth != 0)).sum() / div * 100

    return comparison, percentSuccessful, percentUndecected, percentOver
    

def compareWithWorm(wormBody, crop, id, binaryNumber):
    thisWorm = np.zeros((crop[3], crop[2]), dtype=np.uint8)
    for p in wormBody: thisWorm[p] = 255
    if binaryNumber is not None:
        matchedWorm = cv2.imread("EachWorm/" + id + "_" + ("0" + str(binaryNumber) if binaryNumber < 10 else str(binaryNumber)) + "_ground_truth.png", 0)
        matchedWorm = cropImage(matchedWorm, crop)
    else: matchedWorm = np.zeros((crop[3], crop[2]), dtype=np.uint8)

    comparison, percentSuccessful, percentUndecected, percentOver = createComparisonImage(thisWorm, matchedWorm)

    _, thresh = cv2.threshold(cv2.cvtColor(comparison, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)
    x, y, w, h = cv2.boundingRect(thresh)
    imgDim = 200
    w = min(w, imgDim)
    h = min(h, imgDim)

    strip = np.zeros((300, 600, 3), dtype=np.uint8)
    xpos = 400 + int((imgDim - w)/2)
    ypos = int((imgDim - h)/2)
    strip[ypos:ypos+h, xpos:xpos+w, ::] = comparison[y:y+h, x:x+w, ::]

    if binaryNumber is not None:
        cv2.putText(strip, "Percent matched: " + "{0:.2f}".format(percentSuccessful) + "%", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255, 0))
        cv2.putText(strip, "Excess matched: " + "{0:.2f}".format(percentOver) + "%", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(255, 0, 0))
        cv2.putText(strip, "Percent undetected: " + "{0:.2f}".format(percentUndecected) + "%", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255))
    else: cv2.putText(strip, "No matching worm found.", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255))

    return strip

def matchWorms(wormBodies, id, crop):
    #I would just use a normal array for this, but some worms are missing (IE D15 is missing worm number 1)
    eachworm = {}
    filenames = glob.glob("EachWorm/" + id + "*")

    actualWormNum = len(filenames)

    if actualWormNum == 0:
        print("- !!! Failed to load any worms for this ID !!!")
        print("- Please make sure the EachWorm folder is properly populated.")

    ourWormNum = len(wormBodies)
    print("- Found", ourWormNum, "worms. In this image, there are", actualWormNum, "worms.")

    idmatch = re.compile(r"_(\d{2})_")
    for filename in filenames: #imread each foreground worm and put it in the dict above
        readWorm = cv2.imread(filename, 0)
        readWorm = cropImage(readWorm, crop)
        idsearch = idmatch.search(filename)
        number = idsearch.groups(1)
        number = int(number[0])
        eachworm[number] = readWorm

    actMatching = {v : None for v in eachworm.keys()}
    change = True
    while change:
        change = False
        for wormI in range(len(wormBodies)):
            thisWorm = np.zeros((crop[3], crop[2]), dtype=np.uint8)
            for p in wormBodies[wormI]: thisWorm[p] = 255

            maxCross = 20 #must match 20 pixels to even be considered
            for testNum, testWorm in eachworm.items():
                score = np.count_nonzero(np.bitwise_and(thisWorm, testWorm))
                if score > maxCross: #best fit so far, see if it's matched to a previous worm
                    previousCross = actMatching[testNum]
                    if previousCross is None or previousCross[1] < score:
                        maxCross = score
                        actMatching[testNum] = [wormI, score]
                        change = True
    #put together the final matching
    foundMatching = [None for i in range(ourWormNum)]
    for i in actMatching.keys():
        if actMatching[i] is not None:
            worm, score = actMatching[i]
            foundMatching[worm] = i

    #all worms are matched, print stats and return matching
    unmatchedActualWorms = sum(x == None for x in actMatching.values())
    unmatchedFoundWorms = foundMatching.count(None)
    print("- Failed to find", unmatchedActualWorms , "worms. Found", unmatchedFoundWorms, "nonexisting worms.")
    return foundMatching