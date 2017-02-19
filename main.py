""" HOW TO USE
Before running this program, extract the individual foreground worms are extracted into the EachWorm folder, the total foreground in
the Foreground folder, and the actual image data into the Images folder.
This folder should look like

*.py
>EachWorm
-...
-B02_14_ground_truth.png
-B02_15_ground_truth.png
-...
>Images
-...
-1649_1109_0003_Amp5-1_B_20070424_B02_w1_D9B4F786-FF1E-41D3-ADB9-82C31C6EEBCB.tif
-1649_1109_0003_Amp5-1_B_20070424_B02_w2_CF9DB35A-CEDF-4662-A866-B7093A36E6F6.tif
-...
>Foreground
-...
-B02_binary.png
-...

This program uses Python 3.
The program is handled by the 'render' function . Just running this file (F5) should take care of everything.
When starting the program you will be asked if you want to watch the actual processing. It's faster if you choose not to.
Either way, after the image is processed you will be shown the separated worms in one window and the analysis/comparison in the other.
Use z/x to scroll through detected worms and n/m to scroll between pictures. Use g to go to a new location.
Various information and statistics are printed to the Python environment during the progress of the program.
Some images are missing channels (C13/14). In this case you will be alerted in the console and asked if you want to increment/decrement to the next image.
"""
import cv2
from ReadAndThreshold import loadImage, threshold
from FindSkeleton import getSkel
from Nodes import getNodes, drawNodes
from WormSearch import wormSearch, showWorms
from FloodWorms import floodWorms
from Comparison import *
from Settings import settings
from DeadWormDetector import detectDeadWorms
import numpy as np

class IDManager:
    """
    Simple class that manages which ID we're looking at and increments/decrements
    """
    currentL = 0
    currentI = 3

    @staticmethod
    def getID():
        return chr(IDManager.currentL+65) + format(IDManager.currentI, "02")

    @staticmethod
    def incID():
        if IDManager.currentI >= 24:
            if IDManager.currentL == 4:
                return False
            else:
                IDManager.currentI = 1
                IDManager.currentL += 1
        else: IDManager.currentI += 1

        return True

    @staticmethod
    def decID():
        if IDManager.currentI == 1:
            if IDManager.currentL == 0:
                return False
            else:
                IDManager.currentI = 24
                IDManager.currentL -= 1
        else:
            IDManager.currentI -= 1
        return True

    @staticmethod
    def parseID(instring):
        instring.replace(' ', '')
        letter = ord(instring[0].upper())
        if letter < 65 or letter > 69: raise ValueError()
        number = int(instring[1:]) #raises valueerror for us
        if number < 0 or number > 24 or (letter == 69 and number > 4): raise ValueError()
        IDManager.currentL = letter - 65
        IDManager.currentI = number

    @staticmethod
    def inputID(string = "Which location would you like to go to? "):
        responseDone = False
        while not responseDone:
            response = input(string)
            if response != "":
                try:
                    IDManager.parseID(response)
                except ValueError:
                    print("That wasn't a valid location.")
                    continue
            responseDone = True

def loadData(ID):
    """
    Gets the relevant data needed for worm searching etc
    """
    pair = loadImage(ID)
    print("Thresholding... (~10 seconds)")
    binary, crop = threshold(pair)

    cx, cy, cw, ch = crop
    pair[1] = pair[1][cy:cy + ch, cx:cx + cw]

    print("Finding skeleton... (~5 seconds)")
    skel = getSkel(binary)

    return skel, binary, pair[1], crop

def render():
    print("Use z/x to scroll through segmented worms, n/m to process the previous/next image, g to type in a new ID, and q to quit.")

    IDManager.inputID("Would you like to start at a certain well/row ? (i.e. 'a4', 'B08', or empty for 'any') ")
    response = input("Do you want to be shown the process? (y/n) ").lower()
    settings.vid = response[0] == 'y'

    while True:
        currentID = IDManager.getID()
        cv2.destroyAllWindows()
        print("*** Processing" , currentID, "***" )
        try:
            skel, binary, img, crop = loadData(currentID)
            print("Finding endpoints...")
            nodes = getNodes(skel)

            print("Running graph search...")
            worms, sharedEdges = wormSearch(skel, nodes)

            #hasn't happened but is possible
            if len(worms) == 0:
                cv2.destroyAllWindows()
                print("!!! Failed to find any worms for this image !!!")
                response = input("Decrement, increment, or go to new location? (d/i/g) ").lower()
                if response[0] == 'g':
                    IDManager.inputID()
                elif response[0] == 'd':
                    IDManager.decID()
                else:
                    IDManager.incID()
                continue

            print("Reigon filling...")
            wormBodies, dirs, thicknesses = floodWorms(binary, worms, sharedEdges)
            print("Finished processing.")
            cv2.destroyAllWindows()
            print("*** Comparing results with ground truth data ***")
            print("Comparing binary image...")
            foreground = cv2.imread("Foreground/" + currentID + "_binary.png", 0)
            bcomparison = None
            if foreground is None:
                print("- !!! Could not find the foreground image for this ID !!!")
                print("- Please make sure the foreground folder is extracted to the Foreground directory in the same directory as this file.")
            else:
                x,y,w,h = crop
                foreground = foreground[y:y+h,x:x+w]
                bcomparison, percentSuccessful, percentUndecected, percentOver = createComparisonImage(cv2.dilate(binary, np.ones((3,3))), foreground)
                print("- Matched " + "{0:.2f}".format(percentSuccessful) + "% of the foreground.")
                print("- Matched " + "{0:.2f}".format(percentUndecected) + "% more than the foreground")
                print("- Left " + "{0:.2f}".format(percentUndecected) + "% of the foreground unmatched.")
            print("Comparing each worm...")
            matching = matchWorms(wormBodies, currentID, crop)

            print("*** Detecting dead worms  ***")
            ranges, stdDevs, textureLQRs, scores = detectDeadWorms(wormBodies, dirs, thicknesses, img)

            print("*** Generating colour map ***")
            num = len(wormBodies)
            mul = 255 / num
            map = np.zeros(binary.shape, dtype=np.uint8)
            for i in range(len(wormBodies)):
                worm = wormBodies[i]
                for p in worm: map[p] = (i+1) * mul
            map = cv2.applyColorMap(map, cv2.COLORMAP_JET)

            if bcomparison is not None: map = np.hstack((map, bcomparison))
            cv2.imshow("Segmentation of " + currentID, map)

            print("Done.")
            iterator = 0
            three = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            while True:
                iterator %= num
                drawing = three.copy()
                for p in wormBodies[iterator]: drawing[p] = (0, 0, 255)
                out = cv2.addWeighted(three, 0.5, drawing, 0.5, 0)

                comparison = compareWithWorm(wormBodies[iterator], crop, currentID, matching[iterator])

                dirRange = ranges[iterator]
                if dirRange > 0.65: colour = (0,0,255)
                elif dirRange > 0.5: colour = (0,165,255)
                else: colour = (0,255,0)
                cv2.putText(comparison, "Direction range: " + "{0:.2f}".format(dirRange), (10, 180),cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour)

                stdDev = stdDevs[iterator]
                colour = (0,255, 0) if stdDev < 0.2 else (0,0,255)
                cv2.putText(comparison, "Direction std. dev: " + "{0:.2f}".format(stdDev), (10, 210),cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour)

                lqr = textureLQRs[iterator]
                if lqr < 10: colour = (0,0,255)
                elif lqr < 30: colour = (0,165,255)
                else: colour = (0,255,0)
                cv2.putText(comparison, "Texture LQR: " + str(lqr), (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour)

                thickness = thicknesses[iterator]
                if thickness > 10: colour = (0,255,0)
                elif thickness > 5: colour = (0,165,255)
                else: colour = (0,0,255)
                cv2.putText(comparison, "Average thickness: " + "{0:.2f}".format(thickness), (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour)

                if scores[iterator] >= 100:
                    cv2.putText(comparison, "Dead", (450, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255))
                else: cv2.putText(comparison, "Alive", (450, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0))

                cv2.imshow("Individual worms in " + currentID, out)
                cv2.imshow("Comparison/Analysis of " + currentID, comparison)

                ch = cv2.waitKey()
                if ch == ord('q'):
                    cv2.destroyAllWindows()
                    exit()
                elif ch == ord('x'):
                    iterator += 1
                elif ch == ord('z'):
                    iterator -= 1
                elif ch == ord('m'):
                    cv2.destroyAllWindows()
                    IDManager.incID()
                    break
                elif ch == ord('n'):
                    cv2.destroyAllWindows()
                    IDManager.decID()
                    break
                elif ch == ord('g'):
                    cv2.destroyAllWindows()
                    IDManager.inputID()
                    break

        except FileNotFoundError:
            cv2.destroyAllWindows()
            print("!!! Failed to load images for this worm !!!")
            response = input("Decrement, increment, or go to new location? (d/i/g) ").lower()
            if response[0] == 'g':
                IDManager.inputID()
            elif response[0] == 'd':
                IDManager.decID()
            else:
                IDManager.incID()

render()
