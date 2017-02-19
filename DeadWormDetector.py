import math
import numpy as np

def detectDeadWorms(wormBodies, dirs, thicknesses, img):
    #first find the LQR of pixel intensities under each worm.
    lqrs = []
    for worm in wormBodies:
        hist = [0 for i in range(256)]
        for y,x in worm: hist[img[y,x]] += 1 #just count the number of intensities
        num = len(worm)
        first = num / 4
        second = 2 * first
        count = bin = 0
        while count < first:
            count += hist[bin]
            bin += 1
        firstC = bin
        while count < second:
            count += hist[bin]
            bin += 1
        lqrs.append(bin - firstC)
    # For directions, the higher the range and stddev, the straighter (dead) the worm.
    # first find the ranges and standard deviations.
    ranges = [np.amax(dir) - np.amin(dir) for dir in dirs]
    stdDevs = [np.std(dir) for dir in dirs]

    scores = []
    deadWorms = 0

    for i in range(len(wormBodies)):
        #almost surely dead
        if ranges[i] > 0.8 or thicknesses[i] <= 8.2:
            scores.append(100)
            deadWorms += 1

        #almost surely alive
        elif ranges[i] < 0.3 or stdDevs[i] < 0.15 or lqrs[i] > 25: scores.append(0)

        #probably dead
        elif ranges[i] > 0.65 or stdDevs[i] > 0.21 or thicknesses[i] < 6:
            scores.append(100)
            deadWorms += 1

        #probably alive
        elif ranges[i] < 0.25 or stdDevs[i] < 0.1 or lqrs[i] > 40:
            scores.append(0)

        else: #just make our best guess
            score = 0
            score += ranges[i] * 50
            if ranges[i] > 0.6: score *= 2.1
            if stdDevs[i] < 0.19: score += stdDevs[i] * 100
            else: score += stdDevs[i] * 150
            if lqrs[i] < 20: score += 20 - lqrs[i]
            else: score -= (lqrs[i] - 20) ** 2
            scores.append(score)
            if score >= 100: deadWorms += 1

    print("Found", deadWorms, "dead worms.")

    return ranges, stdDevs, lqrs, scores
