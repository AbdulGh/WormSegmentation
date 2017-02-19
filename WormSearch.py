import cv2
import numpy as np
import Tools as nt
from enum import Enum
from operator import itemgetter
from Settings import settings
import math

"""
There is quite a lot in this file. It turns the skeleton into a graph and then creates worms at the endpoint that
grow greedily and take edges that are most similar to the edges they have taken so far.

The 'nodes' are endpoints and bifurcations. Bifurcations refer to splits in the skeleton.
The 'edges' are segments connecting nodes in the skeleton, along with directional data.

Direction is used to record the shape of worms, when growing a worm we can co-relate potential segments with
the worm so far and see which one is the best match.
"""


"""
utility class for storing the direction a search is heading in
(as a worm is unlikely to have sharp bends)

Indicies refer to the order returned by functions in NeighbourTools, i.e
0       1   2     3   4    5  6      7
NW, N, NE, E, SE, S, SW, W
"""
class Direction(Enum):
    NW, N, NE, E, SE, S, SW, W = range(8)

    @staticmethod
    def getDirection(fromP, toP):
        xdiff = toP[1] - fromP[1]
        ydiff = toP[0] - fromP[0]

        direction = None
        if xdiff == -1:
            if ydiff == -1:
                direction = Direction.NW
            elif ydiff == 0:
                direction = Direction.W
            elif ydiff == 1:
                direction = Direction.SW
        elif xdiff == 0:
            if ydiff == -1:
                direction = Direction.N
            elif ydiff == 1:
                direction = Direction.S
        if xdiff == 1:
            if ydiff == -1:
                direction = Direction.NE
            elif ydiff == 0:
                direction = Direction.E
            elif ydiff == 1:
                direction = Direction.SE

        if direction is None: raise Exception("Points are not adjacent")
        return direction

    @staticmethod
    def getDirections(segment):
        """counts the directions in a segment of the worm"""
        segmentDirs = np.zeros(len(Direction))
        for i in range(len(segment) - 1):
            first = segment[i]
            second = segment[i + 1]
            direction = Direction.getDirection(first, second)
            if direction is None: raise Exception("Disjoint path in getDirections")
            segmentDirs[direction.value] += 1
        #normalize, so we get fractions of time spent heading in each direction
        segmentDirs /= len(segment)
        return segmentDirs

    @staticmethod
    def getRelDirection(p1, p2):
        """returns the general direction from the first point to the second"""
        dy = p1[0] - p2[0]
        dx = p2[1] - p1[1]
        angle = np.arctan2(dy, dx)
        sectionSize =  np.pi / 8
        section = angle // sectionSize
        if section == 0 or section == 15: return Direction.E

        section = (section - 1) // 2
        index = (3 - section) % 8
        return Direction(index)

    
    def getDistance(self, i2):
        """returns the angle between this direction and a neighbour index"""
        i1 = self.value
        x = (i1 - i2) % 8
        y = (i2 - i1) % 8
        return min(x, y)

    
    def move(self, coords):
        """translates some coordinates along this direction"""
        x, y = coords

        if self == Direction.N:
            return (x, y - 1)
        elif self == Direction.NE:
            return (x + 1, y - 1)
        elif self == Direction.E:
            return (x + 1, y)
        elif self == Direction.SE:
            return (x + 1, y + 1)
        elif self == Direction.S:
            return (x, y + 1)
        elif self == Direction.SW:
            return (x - 1, y + 1)
        elif self == Direction.W:
            return (x - 1, y)
        else:
            return (x - 1, y - 1)


def dirCorr(d1, d2):
    """
    used to see how similar two "direction distributions" are
    """
    score = 0
    for shift in range(-2, 3):
        mul =  (3 - abs(shift)) / 3
        shifted = np.roll(d2, shift)
        score += np.corrcoef(d1, shifted)[0,1] * mul
    return score

class Edge:
    """
    stores the edge between two 'nodes' (endpoints or bifurcations). Co-responds to a worm segment
    """
    def __init__(self, fromN, to, dirs, points):
        self.fromN = fromN
        self.to = to
        self.directions = dirs
        self.points = points
        self.owner = None
        self.betweenBifs = False
        self.used = 0

    def getDirs(self, fromN):
        #you can reverse the directions with a shift of 4 if needed
        if fromN == self.fromN: return self.directions
        else: return np.roll(self.directions, 4)

    def getOtherNode(self, node):
        if node == self.fromN: return self.to
        elif node == self.to: return self.fromN
        else: raise Exception("This node does not exist in this edge")

    def equal(self, node1, node2):
        """returns 1 if it's the same edge, -1 if it's reversed, 0 otherwise"""
        if node1 == self.fromN and node2 == self.to: return 1
        if node1 == self.to and node2 == self.fromN: return -1
        return 0

minWormLength = 60
maxWormLength = 170
minEdgeLen = 10
class Worm:
    """
    Manages adding/removing segments to a worm
    Also generates likelihood scores for potential segments
    """
    def __init__(self, start):
        self.nodes = [start]
        self.head = start
        start.owner = self
        self.edges = []
        self.tabu = []
        self.length = 0
        self.dead = False
        self.dirs = np.zeros(len(Direction))

    def swallowEdge(self, edge):
        if edge.owner != None: raise Exception("This edge is owned by some other worm")
        if edge in self.tabu: raise Exception("Can't take this edge!")
        self.tabu.append(edge)
        self.length += len(edge.points)
        self.dirs += edge.getDirs(self.head)
        self.edges.append(edge)
        self.head = edge.getOtherNode(self.head)
        self.nodes.append(self.head)
        if not edge.betweenBifs: edge.owner = self
        else: edge.used += 1

    def loseEdge(self, edge):
        if edge not in self.edges: raise Exception("I don't own this edge")

        self.length = 0
        self.head = self.nodes[0]
        self.nodes=[self.head]
        self.dirs = np.zeros(len(Direction))
        remEdges = []
        i = 0
        #count which edges we still have
        while i < len(self.edges) and self.edges[i] != edge:
            self.dirs += edge.getDirs(self.head)
            self.head = self.edges[i].getOtherNode(self.head)
            self.nodes.append(self.head)
            remEdges.append(self.edges[i])
            self.length += len(self.edges[i].points)
            i += 1
        #revoke ownership over the rest
        while i < len(self.edges):
            self.edges[i].owner = None
            i += 1
        self.edges = remEdges
        if len(self.edges) == 0: self.dead = True #someone ate us

    def checkAgreement(self, edge):
        """
        checks how much a given edge agrees with the general direction of this worm
        """
        if edge in self.edges: raise Exception("Already own this edge!")
        elif len(edge.points) + self.length > maxWormLength: return False
        elif self.length == 0: return 4 #hungry

        wormFracs = self.dirs / self.length
        edgeFracs = edge.getDirs(self.head)/len(edge.points)
        return dirCorr(wormFracs, edgeFracs)

    def checkSelfAgreement(self, edge):
        """
        checks how much a segment of this worm agrees with the rest
        used when another worm wants this segment and we have to fight about it
        """
        if edge not in self.edges:
            raise Exception("I don't own this edge!")
        consideringLen = len(edge.points)
        consideringDirs = None
        totalDirs = np.zeros(len(Direction))
        for n in range(len(self.nodes)-1):
            thisNode = self.nodes[n]
            if self.edges[n] == edge:
                consideringDirs = edge.getDirs(thisNode) / consideringLen
            else: totalDirs += self.edges[n].getDirs(thisNode)
        totalDirs /= self.length - consideringLen
        return dirCorr(consideringDirs, totalDirs)

def wormSearch(skel, nodes):
    finishedWorms = []
    edgeList = {}
    edgeReferences = []

    y,x = skel.shape
    #used for fast lookup of ownership
    skelMap = [[False for i in range(x)] for j in range(y)]
    bifurcations = []
    for node in nodes:
        x,y = node.coords
        skelMap[y][x] = node
        if not node.endpoint:
            edgeList[node] = []
            bifurcations.append(node)

    #first we see which nodes are connected to each other
    #start with a 'multi-DFS' from bifurcations to find worm clusters
    #if bifurcations are close together we may just merge them
    nearbyBifLen = 25
    toMerge = []
    for node in bifurcations:
        x, y = node.coords
        #each fringe is a tuple, (coords, current direction, history of directions, path)
        frontier = []
        neighbours = nt.getNeighbourList(x, y, skel)
        nonzero = np.nonzero(neighbours)[0]
        for n in nonzero:
            direction = Direction(n)
            nx, ny = direction.move((x,y))
            dirs = {direction:0 for direction in Direction}
            dirs[direction] += 1
            frontier.append(((nx, ny), direction, dirs, [(nx,ny)]))

        while len(frontier) != 0:
            if settings.vid:
                frame = cv2.cvtColor(skel, cv2.COLOR_GRAY2BGR)
                for fringe in frontier:
                    for x,y in fringe[3]:
                        frame[y,x] = (255,0,0)
                    fx, fy = fringe[3][0]
                    frame[fy, fx] = (0,0,255)
                    lx, ly = fringe[3][-1]
                    frame[ly, lx] = (0, 255, 0)

                for n in nodes:
                    if n.cluster:
                        cv2.circle(frame, n.coords, 5, (0,255,0))
                cv2.imshow("Processing...", frame)
                cv2.waitKey(1)

            newFrontier = []
            for fringe in frontier:
                fcoords, direction, dirs, path = fringe
                fx, fy = fcoords
                if skel[fy,fx] == 0: raise Exception("Searching outside of the skeleton")

                if skelMap[fy][fx] == True: continue #already been explored, let this fringe die
                elif skelMap[fy][fx] == False: #pick the least disruptive neighbour and go
                    skelMap[fy][fx] = True
                    neighbours = nt.getNeighbourList(fx, fy, skel)
                    nonzero = np.nonzero(neighbours)[0]
                    best = min(nonzero, key=lambda p: direction.getDistance(p))
                    direction = Direction(best)
                    dirs[direction] += 1
                    ncoords = direction.move(fcoords)
                    path.append(ncoords)
                    newFrontier.append((ncoords, direction, dirs, path))
                else: #hit another node, build the edge and add it to the edge lists
                    other = skelMap[fy][fx]
                    if len(path) < minEdgeLen and other.endpoint: # assume it's an artifact left over from skeletonization
                        other.done = True
                        continue
                    elif len(path) > maxWormLength: #split it up into even length worms
                        other.done = True
                        estimatedWormNum = math.ceil(len(path) / maxWormLength)
                        estimatedWormLength = int(len(path) / estimatedWormNum)
                        for i in range(estimatedWormNum):
                            thisSegment = path[i * estimatedWormLength: (i + 1) * estimatedWormLength]
                            finishedWorms.append((thisSegment, Direction.getDirections(thisSegment)))
                        continue
                    other.cluster = True
                    finalDirs = np.array([dirs[d] for d in Direction])
                    edge = Edge(node, other, finalDirs, path)
                    if other != node:
                        if other.endpoint:
                            if other in edgeList:
                                raise Exception("Endpoints should be connected to one thing only")
                            else:
                                edgeList[node].append(edge)
                                edgeList[other] = [edge]
                                edgeReferences.append(edge)
                        else:
                            if len(path) < nearbyBifLen \
                                and not(all(it2 in checkedList for it2 in [node, other]) for checkedList in toMerge):
                                    toMerge.append([node, other, edge])
                            else:
                                edge.betweenBifs = True
                                edgeList[node].append(edge)
                                edgeList[other].append(edge)
                                edgeReferences.append(edge)
            frontier = newFrontier

    #merge together nearby bifurcations
    #we merge first into second
    while len(toMerge) != 0:
        first, second, connEdge = toMerge.pop()
        for incoming in edgeList[first]:
            incoming.points += connEdge.points
            if settings.vid:
                frame = cv2.cvtColor(skel, cv2.COLOR_GRAY2BGR)
                for px, py in incoming.points: frame[py, px] = (0,255,0)
                cv2.imshow("Processing...", frame)
                cv2.waitKey(1)
            outDirs = connEdge.getDirs(first)
            startNode = incoming.getOtherNode(first)
            incoming.to = second
            if incoming.fromN == startNode:
                incoming.directions += outDirs
            else: #flip edge
                incoming.directions = np.roll(incoming.directions, 4) + outDirs
                incoming.fromN = startNode
            edgeList[second].append(incoming)
        del(edgeList[first])
        bifurcations.remove(first)

        avgX = int( (first.coords[0] + second.coords[0]) / 2)
        avgY = int((first.coords[1] + second.coords[1]) / 2)
        second.coords = (avgX, avgY)

    #after removing small edges and merging birfurcations, we may have some birfurcations connected to only one or two endpoints
    if settings.vid: frame = cv2.cvtColor(skel, cv2.COLOR_GRAY2BGR)
    for node in bifurcations:
        edges = edgeList[node]
        condition1 = len(edges) == 1 and edges[0].getOtherNode(node).endpoint
        condition2 =  len(edges) == 2 and len(edges[0].points) + len(edges[1].points) < maxWormLength \
                                and edges[0].getOtherNode(node).endpoint and edges[1].getOtherNode(node).endpoint
        if condition1 or condition2:
            bifurcations.remove(node)
            node.done = True
            node.cluster = False
            path = dirs = None
            if len(edges) == 1:
                other = edges[0].getOtherNode(node)
                other.done = True
                other.cluster = False
                del(edgeList[other])
                path = edges[0].points
                dirs = edges[0].directions / len(path)
                edges[0].owner = path
            elif len(edges) == 2:
                startE = edges[0]
                endE = edges[1]
                startN = startE.getOtherNode(node)
                endN = endE.getOtherNode(node)
                del (edgeList[startN])
                del (edgeList[endN])
                path = startE.points + endE.points
                startE.owner = endE.owner = path
                dirs = (startE.getDirs(startN) + endE.getDirs(node)).astype(np.float32)
                dirs /= len(path)
                startN.done = endN.done = True
                startN.cluster = endN.cluster = False
            finishedWorms.append((path, dirs))
            if settings.vid:
                for px, py in path:
                    frame[py,px] = (0,255,0)
                    cv2.imshow("Processing...", frame)
                cv2.waitKey(1)
            del(edgeList[node])
        elif len(edges) == 1: node.endpoint = True

    #We now do a normal DFS from the remaining endpoints
    #We think these endpoints are not part of any clusters and so should be connected to one other endpoint
    #If this 'worm' is found to be too long we split it up
    for node in nodes:
        if node.done or node.cluster: continue
        x,y = node.coords
        segment = [node.coords]
        #make the first move 'manually' to set the direction
        neighbours = nt.getNeighbourList(x,y,skel)
        nonzero = np.nonzero(neighbours)[0]
        if len(nonzero) == 0: #standalone pixel, get rid of it
            skel[y,x] = 0
            node.done = True
            continue
        next = nonzero[0]
        direction = Direction(next)
        ix, iy = direction.move((x,y))

        if settings.vid:
            frame = cv2.cvtColor(skel, cv2.COLOR_GRAY2BGR)

        while not node.done:
            segment.append((ix,iy))

            if settings.vid and len(segment) >= 2:
                nx,ny = segment[-2]
                frame[ny, nx] = (255, 0, 0)
                nt.fastSetBGR(frame, (ny,nx), (255, 0, 0))
                nx, ny = node.coords
                nt.fastSetBGR(frame, (ny, nx), (0, 255, 0))
                nt.fastSetBGR(frame, (iy, ix), (0, 255, 0))

                cv2.imshow("Processing...", frame)
                cv2.waitKey(1)

            neighbours = nt.getNeighbourList(ix, iy, skel)
            nonzero = np.nonzero(neighbours)[0]

            #hit the end of this line, find the co-responding endpoint
            if len(nonzero) == 1:
                endpoint = skelMap[iy][ix]

                if endpoint == True or endpoint == False:
                    raise Exception("Couldn't find the endpoint for the end of this path")
                node.done = True
                endpoint.done = True
                if len(segment) >= minWormLength:
                    estimatedWormNum = math.ceil(len(segment)/maxWormLength)
                    estimatedWormLength = int(len(segment)/estimatedWormNum)
                    for i in range(estimatedWormNum):
                        thisSegment = segment[i*estimatedWormLength: (i+1)*estimatedWormLength]
                        finishedWorms.append((thisSegment, Direction.getDirections(thisSegment)))

            else:
                #get the neighbour closest to our direction
                best = min(nonzero, key= lambda p: direction.getDistance(p))
                if direction.getDistance(best) > 2: #sharp bend implies new worm
                    estimatedWormNum = math.ceil(len(segment) / maxWormLength)
                    estimatedWormLength = int(len(segment) / estimatedWormNum)
                    for i in range(estimatedWormNum):
                        thisSegment = segment[i * estimatedWormLength: (i + 1) * estimatedWormLength]
                        finishedWorms.append((thisSegment, Direction.getDirections(thisSegment)))
                    segment = []
                direction = Direction(best)
                ix, iy = direction.move((ix, iy))

    #We now try and find worms out of clusters
    clusterEnds = [c for c in nodes if c.cluster and c.endpoint]
    if len(clusterEnds) > 0:
        #growing stores the worms that are still swallowing edges
        growing = [Worm(end) for end in clusterEnds]
        wormReferences = growing.copy()
        while len(growing) != 0:
            newGrowing = []
            for worm in growing:
                if worm.dead: continue
                head = worm.head

                #see which edges we can swallow
                potential = edgeList[head]
                #make a list w/ co-responding scores
                preferenceList = []
                for edge in potential:
                    if edge.owner == worm or edge in worm.tabu: continue
                    score = worm.checkAgreement(edge)
                    if score is not False: preferenceList.append((edge, score))
                preferenceList.sort(key=itemgetter(1), reverse=True)

                if settings.vid:
                    frame = cv2.cvtColor(skel, cv2.COLOR_GRAY2BGR)
                    for edge in worm.edges:
                        for x, y in edge.points:
                            frame[y, x] = (255, 0, 0)
                    for pair in preferenceList:
                        for x, y in pair[0].points:
                            frame[y, x] = (0, 0, 255)
                    cv2.imshow("Processing...", frame)
                    cv2.waitKey(1)

                while len(preferenceList) != 0:
                    desiredEdge, myScore = preferenceList.pop(0)

                    if settings.vid:
                        frame2 = frame.copy()
                        for x, y in desiredEdge.points:
                            frame2[y, x] = (255, 255, 0)
                        cv2.imshow("Processing...", frame2)
                        cv2.waitKey(1)

                    #if we want the body of another worm
                    if desiredEdge.owner is not None:
                        them = desiredEdge.owner
                        if len(them.edges) == 1: #consume this worm and stop (reached an endpoint)
                            them.loseEdge(desiredEdge)
                            worm.swallowEdge(desiredEdge)
                            break
                        else:
                            theirScore = them.checkSelfAgreement(desiredEdge)
                            if theirScore >= myScore: continue #not good enough, check next edge
                            else:
                                them.loseEdge(desiredEdge)
                                worm.swallowEdge(desiredEdge)
                                newGrowing.append(worm)
                                #add the chopped worm to the growing queue so it can check other possible edges
                                newGrowing.append(them)
                                break
                    else: #if the edge is free we just take it
                        worm.swallowEdge(desiredEdge)
                        newGrowing.append(worm)
                        break
            growing = newGrowing

        # all the worms are finished growing, add any unused edges (i,e loops on bifurcations) as worms
        for n in edgeList:
            for e in edgeList[n]:
                if e.owner is None and not e.betweenBifs:
                    new = Worm(n)
                    new.swallowEdge(e)
                    wormReferences.append(new)

        # normalise the directions for each worm
        for worm in wormReferences:
            if worm.length > minWormLength:
                body = []
                # get normalized directions
                dirFracs = np.zeros(len(Direction))
                for i in range(len(worm.nodes) - 1):
                    edge = worm.edges[i]
                    body.extend(edge.points)
                    dirFracs += edge.getDirs(worm.nodes[i])
                dirFracs /= worm.length
                finishedWorms.append((body, dirFracs))

    sharedEdges = [edge.points for edge in edgeReferences if edge.used > 1]

    return finishedWorms, sharedEdges

def showWorms(img, skel, worms):
    if worms is None or len(worms) == 0: raise Exception("No worms")

    three = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    three[skel == 255] = (0, 0, 255)
    iterator = 0

    while True:
        iterator = iterator % len(worms)
        three[skel == 255] = (0, 0, 255)
        worm = worms[iterator][0]
        for x,y in worm: three[y,x] = (0,255,0)
        x, y = worm[0]
        three[y, x] = (255, 0, 0)

        cv2.imshow("worms", three)

        ch = cv2.waitKey()
        if ch == ord('q'):
            cv2.destroyWindow("worms")
            break
        elif ch == 122:
            iterator += 1
        elif ch == 120:
            iterator -= 1











