class DrawInfo:
    x1 = None
    y1 = None
    x2 = None
    y2 = None

    def __init__(self, coordinateArray):
        self.x1 = int(coordinateArray[0])
        self.y1 = int(coordinateArray[1])
        self.x2 = int(coordinateArray[0]) + int(coordinateArray[2])
        self.y2 = int(coordinateArray[1]) + int(coordinateArray[3])

    def __eq__(self, other):
        if isinstance(other, DrawInfo):
            return self.x1 == other.x1 and self.y1 == other.y1
        
        return False
    
    def __hash__(self):
        return id(self)

    def getDrawCoordinates(self):
        return self.x1, self.y1, self.x2, self.y2