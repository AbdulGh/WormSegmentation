import cv2
import numpy as np
import Tools as nt
from Settings import settings

def isEndPoint(neighbours):
    """
    Something is an endpoint if there 6 consecutive neighbours that are empty
    """
    x = 0
    while x < len(neighbours):
        if neighbours[x] == 0:
            i = 1
            while neighbours[(x + i) % len(neighbours)] == 0:
                i += 1
                if i == 6: return True
            x = x + i + 1
        else: x += 1
    return False

def isBifurcation(neighbours, x, y, skel):
    """
    Quickly finds which pixels are reachable with two steps from this one
    (with some special cases taken into account)
    If 3 white pixels are reachable, it's considered a bifurcation
    """
    l = len(neighbours)
    gaps = 0
    for i in range(l):
        if neighbours[i] != 0 and neighbours[(i+1)%l] == 0: gaps += 1
    if gaps < 3: return False

    visited = [[i != 0 and i != 4 and j != 0 and j != 4  for i in range(5)] for j in range(5)]
    neighCoords = nt.getNeighbourCoords(x, y)
    count = 0
    for c in range(1, 9, 2):
        if neighbours[c] != 0:
            cy, cx = neighCoords[c]
            yoff = cy - y
            if yoff != 0:
                vy = cy + yoff
                for check in range(-1, 2):
                    if skel[vy, cx + check] != 0:
                        count += 1
                        visited[y - vy - 2][x - (cx + check) - 2] = True
                        if check != -1: visited[y - vy - 2][x - (cx + check) - 3] = True
                        if check != 1: visited[y - vy - 2][x - (cx + check) - 1] = True
                        break
            else:
                xoff = cx - x
                vx = cx + xoff
                for check in range(-1, 2):
                    if skel[cy + check, vx] != 0:
                        count += 1
                        visited[y - (cy + check) - 2][x - vx - 2] = True
                        if check != -1: visited[y - (cy + check) - 3][x - vx - 2] = True
                        if check != 1: visited[y - (cy + check) - 1][x - vx - 2] = True
                        break

    for c in range(0, 8, 2):
        if neighbours[c] != 0:
            cy, cx = neighCoords[c]
            dx = cx - x
            dy = cy - y
            p1 = (cy + dy, cx + dx)
            p2 = (cy, cx + dx)
            p3 = (cy + dy, cx)

            for p in [p1, p2, p3]:
                py, px = p
                vy, vx = py -y +  2, px - x + 2
                if skel[p] != 0 and not visited[vy][vx]:
                    count += 1
                    break
    return count >= 3

class Node:
    def __init__(self, coords, endpoint=False):
        self.coords = coords
        self.endpoint = endpoint
        self.done = False
        self.cluster = not endpoint

def getNodes(skel):
    h, w = skel.shape
    nodes = []
    for y in range(1, h - 1):
        for x in range(1, w-1):
            if skel[y, x] != 0:
                neighbours = nt.getNeighbourList(x, y, skel)
                if isEndPoint(neighbours): nodes.append(Node((x,y), True))
                elif isBifurcation(neighbours, x, y, skel):
                    bif = Node((x,y), False)
                    nodes.append(bif)

    if settings.vid:
        cv2.imshow("Processing...", drawNodes(skel, nodes))
        cv2.waitKey(100)
    return nodes
def drawNodes(skel, nodes):
    three = cv2.cvtColor(skel, cv2.COLOR_GRAY2BGR)
    three[skel==255] = (255,0,0)
    for node in nodes:
        if node.endpoint: colour = (0,255,0)
        else: colour = (0,0,255)
        coord = node.coords
        three[coord[1], coord[0]] = colour
    return three
