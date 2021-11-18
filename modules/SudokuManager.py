import cv2 as cv
import numpy as np
from numpy.core.numeric import indices

from modules.DrawInfo import DrawInfo
import modules.utils as utils

class SudokuManager:
    '''
    Takes a numpy array of numbers and provides method to interact
    as if it were a sudoku puzzle
    Takes an image of the sudoku puzzle; for image manipulation
    '''
    sudokuPuzzle = None
    sudokuImage = None

    # x,y,w,h,area
    boundingBoxCoordinates = None
    coordinateMap = None


    def __init__(self, puzzle, image):
        self.sudokuPuzzle = puzzle
        self.sudokuImage = image

        # Extract the bounding box locations
        self.boundingBoxCoordinates = self.extractBoundingBoxes()
        self.coordinateMap = self.mapBoundingBoxesToPuzzle(self.boundingBoxCoordinates)

    def printSudoku(self):
        print(f'{self.sudokuPuzzle}')
    
    def extractBoundingBoxes(self):
        '''
        Source of this god-send code
        https://towardsdatascience.com/checkbox-table-cell-detection-using-opencv-python-332c57d25171
        '''

        # Binarize the image
        # Convert to grayscale and apply a threshold setting each pixel to ON or OFF based on
        # a threshold number
        gray_scale = cv.cvtColor(self.sudokuImage, cv.COLOR_BGR2GRAY)
        th, img_bin = cv.threshold(gray_scale, 150, 225, cv.THRESH_BINARY)
        img_bin =~ img_bin

        # This is to avoid the detection of False Positives in text
        line_min_width = 15
        kernal_h = np.ones((1, line_min_width), np.uint8)
        kernal_v = np.ones((line_min_width, 1), np.uint8)

        # Applying horizonal kernal on the Image
        img_bin_h = cv.morphologyEx(img_bin, cv.MORPH_OPEN, kernal_h)

        # Applying Vertical Keranl on the Image
        img_bin_v = cv.morphologyEx(img_bin, cv.MORPH_OPEN, kernal_v)

        # And now Fusion Ha!
        img_bin_final = img_bin_h|img_bin_v

        final_kernal = np.ones((3,3), np.uint8)
        # Apply dilation to handle corner case of unconnected boxes due to lower quality images
        img_bin_final = cv.dilate(img_bin_final, final_kernal, iterations=1)

        # Detect the bounding boxes in the image
        _, labels, stats, _ = cv.connectedComponentsWithStats(~img_bin_final, connectivity=8, ltype=cv.CV_32S)

        stats = stats[2:]
        # Remove the outlier data points
        # Simple outlier removal; order by area, take the last 81 elements
        # Order by area 
        stats = stats[stats[:, 4].argsort()]

        # Take the last 81 elements
        stats = stats[-81:, :]

        # Sort by x coordinates then by y
        stats = stats[np.lexsort((stats[:,0], stats[:,1]))]

        return stats

    def mapBoundingBoxesToPuzzle(self, boundingBoxes):
        '''
        Organizes the bounding boxes to be in the same relative positions as the numbers in memory
        '''
        # Create empty 9 x 9 array
        coordMap = np.empty([9, 9], dtype=object)

        indexOfBoundingBoxes = 0
        for i in range(coordMap.shape[0]):
            for j in range(coordMap.shape[1]):
                # Assign a box info object to it
                coordMap[i,j] = DrawInfo(boundingBoxes[indexOfBoundingBoxes, :])
                indexOfBoundingBoxes += 1

        return coordMap
    

    def highlightNumber(self, numberToHighlight, saveName, isSaving):
        '''
        Highlights all the numbers on the image
        Parameters:
            numberToHighlight: The number that will be highlighted on the image
            saveName: The name that the image will be saved as
            isSaving: A boolean value indicating whether the user wants the image to be saved
        '''
        if numberToHighlight >= 1 and numberToHighlight <= 9:
            sudokuToShow = self.sudokuImage.copy()

            # Find the coordinates of all the numbers to highlight
            indices = np.where(self.sudokuPuzzle == numberToHighlight)
            listOfIndices = list(zip(indices[0], indices[1]))

            for coord in listOfIndices:
                # Draw rectangle on sudoku
                utils.drawRectangle(self.coordinateMap[coord[0], coord[1]], sudokuToShow, utils.Colour.RED.value)

            utils.showAndSave(sudokuToShow, saveName, isSaving)
    

    def highlightNonDrawFor(self, numberToHighlight, saveName, isSaving):
        '''
        Highlights all the places on the sudoku image where a given number can't be placed
        Parameters:
            numberToHighlight: The number that will be tested for placing on the image
            saveName: The name that the image will be saved as
            isSaving: A boolean value indicating whether the user wants the image to be saved
        Returns:
            A truth table indicating where that number could possibly be placed in the puzzle
        '''
        if numberToHighlight >= 1 and numberToHighlight <= 9:
            sudokuToShow = self.sudokuImage.copy()

            # Find coordinates of the instance of number
            indices = np.where(self.sudokuPuzzle == numberToHighlight)
            listOfIndices = list(zip(indices[0], indices[1]))

            # Create a list of coordinates that we must draw on
            dioList = []

            truthTable = np.array(self.sudokuPuzzle, dtype=bool)
            truthTable = np.invert(truthTable)


            for coord in listOfIndices:
                rowCoord = coord[0]
                colCoord = coord[1]

                # Get the row coordinates
                row = self.coordinateMap[rowCoord, :]
                row = row.tolist()

                # Get the column coordinates
                col = self.coordinateMap[:, colCoord]
                col = col.tolist()

                # Get the correct 3 by 3
                startRow = rowCoord - rowCoord % 3
                startCol = colCoord - colCoord % 3
                subBox = []

                for i in range(3):
                    for j in range(3):
                        subBox.append(self.coordinateMap[i + startRow, j + startCol])
                

                dioList.extend(row)
                dioList.extend(col)
                dioList.extend(subBox)

            # Remove duplicates
            dioSet = set(dioList)


            for dio in dioSet:
                # Update truth table (coordinates of drawn image)
                dioCoord = np.where(self.coordinateMap == dio)
                listOfDioCoords = list(zip(dioCoord[0], dioCoord[1]))

                for d in listOfDioCoords:
                    truthTable[d[0], d[1]] = False

            # Find all the areas that are false
            falseCoords = np.where(truthTable == False)
            listOfFalseCoords = list(zip(falseCoords[0], falseCoords[1]))

            for f in listOfFalseCoords:
                # Get appropriate DrawInfo Object
                dio = self.coordinateMap[f[0], f[1]]
                # Draw rectangle on sudoku
                utils.drawRectangle(dio, sudokuToShow, utils.Colour.FUSHIA.value)

            # Show image
            utils.showAndSave(sudokuToShow, saveName, isSaving)

            #print(truthTable)
            return truthTable