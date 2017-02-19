import numpy as np
import cv2
import Tools as nt
from operator import itemgetter
from Settings import settings
from WormSearch import dirCorr

maxCombinedSize = 1800
minSize = 400

def floodWorms(binary, worms, sharedEdges):
    """
    Standard region filling with two additions -
    1) Bifurcations 'spread out' a circle to see which part of the binary image are shared by worms
    2) Bordering regions are noted and merged together if either is too small
    """

    binary = cv2.dilate(binary, np.ones((3,3)))
    h,w = binary.shape
    #black outline
    binary[0,:] = 0
    binary[h-1, :] = 0
    binary[:, 0] = 0
    binary[:, w-1] = 0

    fullWorms = np.zeros(binary.shape, dtype=int)
    #'worms' at the moment are segments of the skeleton paired with the % moving in each direction
    wormDirections = [worm[1] for worm in worms]
    sizes = [0 for i in range(len(worms))]
    lengths = [len(worm[0]) for worm in worms]

    #bifurcations are the segments of the binary worm that are shared.
    counter = len(worms)
    #dict use to add all points in a bifurcation to a worm
    sharedPoints = {}
    # start with region filling just the shared edges. Distance is limited
    for edge in sharedEdges:
        fringe = [((ey,ex), 0) for (ex,ey) in edge]
        allPoints = []
        while len(fringe) != 0:
            (y, x), d = fringe.pop()
            neighbours = nt.getNeighbourCoords(x, y)
            for n in neighbours:
                if binary.item(n) != 0 and fullWorms.item(n) == 0:
                    if d < 8: fringe.append((n,d+1))
                    allPoints.append(n)
                    fullWorms[n] = counter
        sharedPoints[counter] = allPoints.copy()
        counter += 1

    wormBodies = [[] for i in range(len(worms))]
    fringe = {}
    sharedWormMatchings = {shared: [] for shared in sharedPoints.keys()}
    #use the segments of the skeleton as markers
    for i in range(len(worms)):
        for x,y in worms[i][0]:
            if fullWorms[y,x] >= len(worms) and i not in sharedWormMatchings[fullWorms[y,x]]: #in a shared section that has not been noted
                new = sharedPoints[fullWorms[y,x]]
                wormBodies[i].extend(new)
                sizes[i] += len(new)
                sharedWormMatchings[fullWorms[y,x]].append(i)
            else:
                #fullWorms[y,x] = i+1
                fringe[y,x] = i+1
                sizes[i] += 1
                wormBodies[i].append((y,x))

    #used to note which worms are bordering each other and how long that border is
    borders = {}

    if settings.vid:
        frame = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        for j in range(frame.shape[0]):
            for i in range(frame.shape[1]):
                fw = fullWorms[j,i]
                if fw != 0:
                    frame[j,i] = ((fw+1) ** 3 % 255, (fw+1) ** 2 % 255 , fw * 255/len(worms))

    while len(fringe) != 0:
        #each element of the fringe is a coordinate (y,x) matched with the worm currently owning that coordinate
        newFringe = {}
        for y,x in fringe:
            wormIndex = fringe[y,x]
            wormBodies[wormIndex - 1].append((y,x))
            for neighbour in nt.getNeighbourCoords(x,y):
                if binary.item(neighbour) != 0:
                    if fullWorms[neighbour] == 0:
                        fullWorms[neighbour] = wormIndex
                        newFringe[neighbour] = wormIndex

                        # sizes are later used to merge small bordering worms that we think are the same
                        sizes[wormIndex - 1] += 1

                        if settings.vid:
                            nt.fastSetBGR(frame, neighbour, ((wormIndex+1) ** 3 % 255, (wormIndex+1) ** 2 % 255 , (wormIndex) * 255/len(worms)))

                    else: #make a note if a region is touching another
                        neighbour = fullWorms[neighbour]
                        if neighbour == wormIndex or neighbour >= len(worms): continue
                        pair = (min(neighbour, wormIndex) - 1, max(neighbour, wormIndex) - 1)
                        if pair not in borders: borders[pair] = 1
                        else: borders[pair] += 1
        fringe = newFringe
        if settings.vid:
            cv2.imshow("Processing...", frame)
            cv2.waitKey(1)

    # merge bordering worms
    sortedPseudoDict = [(k, v) for k, v in borders.items()]
    sortedPseudoDict.sort(key=itemgetter(1))
    sortedNeighbours = [[p[0][0], p[0][1]] for p in sortedPseudoDict]
    while len(sortedNeighbours) != 0:
        i1, i2 = sortedNeighbours.pop(-1)
        if i1 == i2: continue
        size1 = sizes[i1]
        size2 = sizes[i2]
        if (size1 < minSize or size2 < minSize) and (size1 + size2 < maxCombinedSize):
            # merge worms
            wormBodies[i1].extend(wormBodies[i2])
            del (wormBodies[i2])
            sizes[i1] += size2
            del (sizes[i2])

            for i in range(len(sortedNeighbours)):
                # if either side of the pair is being merged into the first, we change its index accordingly
                if sortedNeighbours[i][0] == i2:
                    sortedNeighbours[i][0] = i1
                elif sortedNeighbours[i][1] == i2:
                    sortedNeighbours[i][1] = i1
                # shift down indicies to fill the hole
                if sortedNeighbours[i][0] > i2: sortedNeighbours[i][0] -= 1
                if sortedNeighbours[i][1] > i2: sortedNeighbours[i][1] -= 1

            # add directions and combine the length (both later used to guess if the worm is dead)
            # assume that each worm segment has similar directions to the one it's merging to
            dirsToMerge = wormDirections[i2]
            dirsToMergeReverse = np.roll(dirsToMerge, 4)
            if dirCorr(wormDirections[i1], dirsToMerge) < dirCorr(wormDirections[i1], dirsToMergeReverse):
                dirsToMerge = dirsToMergeReverse
            # combine and re-normalize
            wormDirections[i1] = wormDirections[i1] * lengths[i1] + dirsToMerge * lengths[i2]
            del (wormDirections[i2])
            lengths[i1] += lengths[i2]
            del (lengths[i2])
            wormDirections[i1] /= lengths[i1]
    # finally, get rid of bodies that are too small to be worms
    finalWorms = []
    finalDirs = []
    finalThicknesses = []
    for i in range(len(wormBodies)):
        if len(wormBodies[i]) > 300:
            finalWorms.append(wormBodies[i])
            finalDirs.append(wormDirections[i])
            finalThicknesses.append(sizes[i]/lengths[i])

    return finalWorms, finalDirs, finalThicknesses

def showFullWorms(img, fullWorms):
    num = np.amax(fullWorms)
    iterator = 0

    while True:
        iterator = iterator % num
        three = img.copy()
        three = cv2.cvtColor(three, cv2.COLOR_GRAY2BGR)
        three[fullWorms == iterator+1] = (0, 0, 255)

        cv2.imshow("FullWorms", three)

        ch = cv2.waitKey()
        if ch == 27:
            cv2.destroyWindow("FullWorms")
            exit()
        elif ch == 122:
            iterator += 1
        elif ch == 120:
            iterator -= 1

