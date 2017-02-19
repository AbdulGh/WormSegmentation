def getNeighbourCoords(x,y, w=None, h =None):
    if w is None: w = x+2
    if h is None: h = y+2
    x1 = x - 1
    x2 = x
    x3 = x + 1
    y1 = y - 1
    y2 = y
    y3 = y + 1

    return [point for point in  [(y1,x1), (y1,x2), (y1,x3), (y2, x3), (y3, x3), (y3, x2), (y3,x1), (y2, x1)] \
            if point[0] >= 0 and point[0] < h and point[1] >= 0 and point[1] < w]

def getNeighbourCount(x, y, img):
    h, w = img.shape
    neighbours = 0
    for j in range(max(y - 1, 0), min(y + 2, h)):
        for i in range(max(x-1,0), min(x+2,w)):
            if i == x and j == y:
                continue
            if img.item(j,i) != 0:
                neighbours += 1
    return neighbours

#gets the ring around the pixel
#assumes we're aren't on the border of the image
def getNeighbourList(x, y, img):
    neighbours = []

    for i in range (x-1, x+2):
        neighbours.append(img.item((y-1,i)))
    neighbours.append(img.item((y,x+1)))
    for i in range (x+1, x-2, -1):
        neighbours.append(img.item(((y+1,i))))
    neighbours.append(img.item(((y, x-1))))

    return neighbours

#itemset is faster when dealing w/ numpy arrays but individually accessing each channel can be a pain
def fastSetBGR(img, coords, BGR):
    y,x = coords
    b,g,r = BGR
    img.itemset((y,x, 0), b)
    img.itemset((y, x, 1), g)
    img.itemset((y, x, 2), r)