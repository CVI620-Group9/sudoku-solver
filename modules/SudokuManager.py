import cv2 as cv
import numpy as np
import copy

from modules.DrawInfo import DrawInfo
import modules.utils as utils

class SudokuManager:
    '''
    Takes a numpy array of numbers and provides method to interact
    as if it were a sudoku puzzle
    Takes an image of the sudoku puzzle; for image manipulation
    '''
    sudokuPuzzle = None  # 2d array
    sudokuImage = None  # Sudoku image;
    possible_value = {}  # array for the possible values of each box in the puzzle
    editImage = None #edited sudoku image for highlights

    # x,y,w,h,area
    boundingBoxCoordinates = None
    coordinateMap = None

    def __init__(self, image):
        self.sudokuImage = image

        # Extract the bounding box locations
        self.boundingBoxCoordinates = self.extractBoundingBoxes()
        self.coordinateMap = self.mapBoundingBoxesToPuzzle(self.boundingBoxCoordinates)

        self.sudokuPuzzle = self.fillPuzzle()

        # initialize list of possible values
        for i in range(1, 10):
            for j in range(1, 10):
                self.possible_value[i, j] = list(range(1, 10))

        # remove list from filled spaces
        for i in range(1, 10):
            for j in range(1, 10):
                if self.sudokuPuzzle[i - 1][j - 1] != 0:
                    self.possible_value[i, j] = []

    def printSudoku(self):
        print(f'{self.sudokuPuzzle}')

    def fillPuzzle(self):
        sudoPuzzle = []
        # Make a copy of the image
        sudokuImageCopy = self.sudokuImage.copy()
        sudokuImageCopy = cv.cvtColor(sudokuImageCopy, cv.COLOR_BGR2GRAY)
        # Iterate over image and slice out number to make a prediction on
        for dInfoRow in self.coordinateMap:
            sudoRow = []
            for dio in dInfoRow:
                x1, y1, x2, y2 = dio.getDrawCoordinates()
                # Get slice of number
                numSlice = sudokuImageCopy[y1:y2, x1:x2]
                prediction = utils.getPredictionForImage(numSlice)
                sudoRow.append(prediction)

            sudoPuzzle.append(sudoRow)

        sudoToReturn = np.array(sudoPuzzle)

        return sudoToReturn

    def extractBoundingBoxes(self):
        '''
        Source of this code
        https://towardsdatascience.com/checkbox-table-cell-detection-using-opencv-python-332c57d25171
        '''

        # Binarize the image
        # Convert to grayscale and apply a threshold setting each pixel to ON or OFF based on
        # a threshold number
        gray_scale = cv.cvtColor(self.sudokuImage, cv.COLOR_BGR2GRAY)
        th, img_bin = cv.threshold(gray_scale, 200, 240, cv.THRESH_BINARY)
        img_bin = ~ img_bin

        # This is to avoid the detection of False Positives in text
        line_min_width = 15
        kernal_h = np.ones((1, line_min_width), np.uint8)
        kernal_v = np.ones((line_min_width, 1), np.uint8)

        # Applying horizonal kernal on the Image
        img_bin_h = cv.morphologyEx(img_bin, cv.MORPH_OPEN, kernal_h)

        # Applying Vertical Keranl on the Image
        img_bin_v = cv.morphologyEx(img_bin, cv.MORPH_OPEN, kernal_v)

        # And now Fusion Ha!
        img_bin_final = img_bin_h | img_bin_v

        final_kernal = np.ones((3, 3), np.uint8)
        # Apply dilation to handle corner case of unconnected boxes due to lower quality images
        img_bin_final = cv.dilate(img_bin_final, final_kernal, iterations=1)

        # Detect the bounding boxes in the image
        _, labels, stats, _ = cv.connectedComponentsWithStats(~img_bin_final, connectivity=8, ltype=cv.CV_32S)

        stats = stats[1:]
        # Remove the outlier data points
        # Simple outlier removal; order by area, take the last 81 elements
        # Order by area 
        stats = stats[stats[:, 4].argsort()]

        # Take the last 81 elements
        stats = stats[-81:, :]

        statsToReturn = []
        # sort on y
        stats = stats[stats[:, 1].argsort()]
        step = 9
        for i in range(0, 81, step):
            # slice out set of 9
            stat_slice = stats[i:i + step, :]
            # sort on x
            stat_slice = stat_slice[stat_slice[:, 0].argsort()]
            statsToReturn.extend(stat_slice)

        statsToReturn = np.array(statsToReturn)

        return statsToReturn

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
                coordMap[i, j] = DrawInfo(boundingBoxes[indexOfBoundingBoxes, :])
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

            # print(truthTable)
            return truthTable

    def drawNumberAt(self, numberToDraw, coordinates, colour):
        '''
        Draws the specified number at the coordinates given
        Colour is a tuple in BGR format indicating the colour that the number should be
        '''
        if self.sudokuPuzzle[coordinates[0], coordinates[1]] == 0:
            print("Adding number: ", numberToDraw)
            # Get matching asset
            numberImage = utils.getNumberImage(numberToDraw, "./assets")

            # Change image colour
            numberImage = utils.changeTextColour(numberImage, colour)
            #print(f"image shape {numberImage.shape}")

            # Get coordinates to fit in the box
            drawCoords = self.coordinateMap[coordinates[0], coordinates[1]]
            x1, y1, x2, y2 = drawCoords.getDrawCoordinates()

            # Calculate the size of the box
            w = x2 - x1
            h = y2 - y1

            # Calculate dimensions that the image should be
            img_w = int(w * 0.50)  # 10 percent margin both sides
            img_h = int(h * 0.50)  # 10 percent margin both sides

            # Resize the image
            numberImage = cv.resize(numberImage, (img_w, img_h))

            # Find the top corner to start from
            centerOfBox_x = x1 + int(w / 2)
            centerOfBox_y = y1 + int(h / 2)

            start_x = centerOfBox_x - int(img_w / 2)
            start_y = centerOfBox_y - int(img_h / 2)

            # Add to image
            self.sudokuImage[start_y:start_y + img_h, start_x:start_x + img_w] = numberImage

            # Add to puzzle in memory (this might change)
            self.sudokuPuzzle[coordinates[0], coordinates[1]] = numberToDraw

            # show
            utils.showAndSave(self.sudokuImage, "Edited sudo", True)
            print(self.sudokuPuzzle)
        else:
            print(f"Cannot place number, theres a(n) {self.sudokuPuzzle[coordinates[0], coordinates[1]]} here!")


    #SOLUTION FUNCTIONS
    #These functions were added here because they cannot be held in a separate file

    def check_row(self, possible_value_, solution_):
        '''
        :param possible_value_: the dict of storing all possible numbers of each cell
        :param solution_: the list of existing solution
        Based on Sudoku rule, check for each row and see if there is an unique possible number across a row.
        If so, update the possible_value_ and solution_.
        '''
        for i in range(1, 10):
            exist = solution_[i - 1]

            for j in range(1, 10):
                possible_value_[i, j] = [x for x in possible_value_[i, j] if x not in exist]

            possible_element = [x for y in [value for key, value in possible_value_.items()
                                            if key[0] == i and len(value) > 0] for x in y]
            unique = [x for x in possible_element if possible_element.count(x) == 1]
            if len(unique) > 0:
                for x in unique:
                    for key, value in {key: value for key, value in possible_value_.items() if
                                       key[0] == i and len(value) > 0}.items():
                        if x in value:
                            self.highlightNumber(x, "highlight", True)
                            self.highlightNonDrawFor(x, "highlight", True)
                            self.drawNumberAt(x, [key[0] - 1, key[1] - 1], utils.Colour.GREEN.value)
                            solution_[key[0] - 1][key[1] - 1] = x
                            possible_value_[key] = []
        return 0

    def check_column(self, possible_value_, solution_):
        """
        :param possible_value_: the dict of storing all possible numbers of each cell
        :param solution_: the list of existing solution
        Based on Sudoku rule, check for each column and see if there is an unique possible number across a column.
        If so, update the possible_value_ and solution_.
        """
        for j in range(1, 10):
            exist = [x[j - 1] for x in solution_]
            for i in range(1, 10):
                possible_value_[i, j] = [x for x in possible_value_[i, j] if x not in exist]

            possible_element = [x for y in [value for key, value in possible_value_.items()
                                            if key[1] == j and len(value) > 0] for x in y]
            unique = [x for x in possible_element if possible_element.count(x) == 1]
            if len(unique) > 0:
                for x in unique:
                    for key, value in {key: value for key, value in possible_value_.items() if
                                       key[1] == j and len(value) > 0}.items():
                        if x in value:
                            self.highlightNumber(x, "highlight", True)
                            self.highlightNonDrawFor(x, "highlight", True)
                            self.drawNumberAt(x, [key[0] - 1, key[1] - 1], utils.Colour.GREEN.value)
                            solution_[key[0] - 1][key[1] - 1] = x
                            possible_value_[key] = []
        return 0

    def box_range(self, number):
        """
        :param number: input the row or column number
        :return: a list of row or column number within the same box
        """
        if number in (1, 2, 3):
            return [1, 2, 3]
        elif number in (4, 5, 6):
            return [4, 5, 6]
        elif number in (7, 8, 9):
            return [7, 8, 9]

    def check_box(self, possible_value_, solution_):
        """
        :param possible_value_: the dict of storing all possible numbers of each cell
        :param solution_: the list of existing solution
        Based on Sudoku, check for each box and see if there is an unique possible number within a box.
        If so, update the possible_value_ and solution_.
        """
        for i in [1, 4, 7]:
            for j in [1, 4, 7]:
                exist = set(
                    [solution_[i_range - 1][j_range - 1] for j_range in range(j, j + 3) for i_range in range(i, i + 3)])
                for k in self.box_range(i):
                    for l in self.box_range(j):
                        possible_value_[k, l] = [b for b in possible_value_[k, l] if b not in exist]

                possible_element = [x for b in [value for key, value in possible_value_.items()
                                                if key[0] in self.box_range(i) and key[1] in self.box_range(j) and len(value) > 0]
                                    for x in b]
                unique = [x for x in possible_element if possible_element.count(x) == 1]
                if len(unique) > 0:
                    for k in unique:
                        for key, value in {key: value for key, value in possible_value_.items()
                                           if key[0] in self.box_range(i) and key[1] in self.box_range(j) and len(
                                value) > 0}.items():
                            if k in value:
                                self.highlightNumber(k, "highlight", True)
                                self.highlightNonDrawFor(k, "highlight", True)
                                self.drawNumberAt(k, [key[0]-1, key[1]-1], utils.Colour.GREEN.value)
                                solution_[key[0] - 1][key[1] - 1] = k
                                possible_value_[key] = []
        return 0

    def check_unique_possible_value(self, possible_value_, solution_):
        """
        :param possible_value_: the dict of storing all possible numbers of each cell
        :param solution_: the list of existing solution
        For each cell, if there is only one possible number, update solution_ and remove from possible_value_
        """
        for key, value in possible_value_.items():
            if len(value) == 1:
                self.highlightNumber(value[0], "highlight", True)
                self.highlightNonDrawFor(value[0], "highlight", True)
                self.drawNumberAt(value[0], [key[0] - 1, key[1] - 1], utils.Colour.GREEN.value)
                solution_[key[0] - 1][key[1] - 1] = value[0]
                possible_value_[key] = []

        return 0

    def loop_basic_rule(self, possible_value_, solution_):
        """
        :param possible_value_: the dict of storing all possible numbers of each cell
        :param solution_: the list of existing solution
        Run all basic rules until there is no update on solution_
        """

        while True:
            checker = True
            solution_old = copy.deepcopy(solution_)
            self.check_row(possible_value_, solution_)
            self.check_column(self.possible_value, self.sudokuPuzzle)
            self.check_box(self.possible_value, self.sudokuPuzzle)
            self.check_unique_possible_value(self.possible_value, self.sudokuPuzzle)

            for i in range(1, 10):
                for j in range(1, 10):
                    if solution_old[i - 1][j - 1] == solution_[i - 1][j - 1]:
                        checker = False
                    else:
                        self.drawNumberAt(solution_[i - 1][j - 1], [i-1, j-1], utils.Colour.GREEN.value)

            if not checker:
                break

    def algorithm(self, possible_list_, possible_value_, solution_):
        """
        :param possible_list_: a markup of a cell
        :param possible_value_: the dict of storing all possible numbers of each cell
        :param solution_: the list of existing solution
        The function applies Crook's algorithm and eliminate impossible numbers.
        If there is any update, re-run all basic rules
        """
        try:
            min_num_possible = min((len(v)) for _, v in possible_list_.items())
            max_num_possible = max((len(v)) for _, v in possible_list_.items())
        except ValueError:
            return 0
        for i in reversed(range(min_num_possible, max_num_possible + 1)):
            for key, value in {key: value for key, value in possible_list_.items() if len(value) == i}.items():
                n_subset = 0
                key_match = set()
                for key_1, value_1 in possible_list_.items():
                    if len(value) < len(value_1):
                        continue
                    else:
                        if set(value_1).issubset(set(value)):
                            key_match.add(key_1)
                            n_subset += 1
                    if n_subset == len(value):
                        for key_2, value_2 in {key: value for key, value in possible_list_.items() if
                                               key not in key_match}.items():
                            possible_value_[key_2] = [x for x in value_2 if x not in value]
                            self.loop_basic_rule(possible_value_, solution_)
        return 0

    def algorithm_row(self, possible_value_, solution_):
        """
        :param possible_value_: the dict of storing all possible numbers of each cell
        :param solution_: the list of existing solution
        Apply Crook's algorithm on each row
        """
        for i in range(1, 10):
            possible_list = {key: value for key, value in possible_value_.items() if key[0] == i and len(value) > 0}
            self.algorithm(possible_list, possible_value_, solution_)
        return 0

    def algorithm_column(self, possible_value_, solution_):
        """
        :param possible_value_: the dict of storing all possible numbers of each cell
        :param solution_: the list of existing solution
        Apply Crook's algorithm on each column
        """
        for j in range(1, 10):
            possible_list = {key: value for key, value in possible_value_.items() if key[1] == j and len(value) > 0}
            self.algorithm(possible_list, possible_value_, solution_)
        return 0

    def algorithm_box(self, possible_value_, solution_):
        """
        :param possible_value_: the dict of storing all possible numbers of each cell
        :param solution_: the list of existing solution
        Apply Crook's algorithm on each box
        """
        for i in [1, 4, 7]:
            for j in [1, 4, 7]:
                possible_list = {key: value for key, value in possible_value_.items() if
                                 key[0] in self.box_range(i) and key[1] in self.box_range(j) and len(value) > 0}
                self.algorithm(possible_list, possible_value_, solution_)
        return 0

    def loop_algorithm(self, possible_value_, solution_):
        """
        :param possible_value_: the dict of storing all possible numbers of each cell
        :param solution_: the list of existing solution
        Apply Crook's algorithm until there is no update on solution_
        """
        while True:
            checker = True
            solution_old = copy.deepcopy(solution_)
            self.algorithm_row(possible_value_, solution_)
            self.algorithm_column(possible_value_, solution_)
            self.algorithm_box(possible_value_, solution_)
            for i in range(1, 10):
                for j in range(1, 10):
                    if solution_old[i - 1][j - 1] == solution_[i - 1][j - 1]:
                        checker = False

            if not checker:
                break
        return 0

    def check_box_eliminate_others(self, possible_value_):
        """
        :param possible_value_: the dict of storing all possible numbers of each cell
        By considering the possible numbers within a box, check if there is a possible number only in one row/column.
        If so, then in the same row/column outside the box, this possible number will be eliminated from all markup.
        """
        for i in [1, 4, 7]:
            for j in [1, 4, 7]:

                possible_element = set([x for b in
                                        [value for key, value in possible_value_.items()
                                         if key[0] in self.box_range(i) and key[1] in self.box_range(j) and len(value) > 0]
                                        for x in b])

                for x in possible_element:
                    available_cell = [key for key, value in possible_value_.items()
                                      if x in value if key[0] in self.box_range(i) and key[1] in self.box_range(j)]
                    if len(set([x[0] for x in available_cell])) == 1:
                        for key in [key for key, value in possible_value_.items()
                                    if key[0] == available_cell[0][0] and key not in available_cell]:
                            possible_value_[key] = [y for y in possible_value_[key] if y != x]

                    if len(set([x[1] for x in available_cell])) == 1:
                        for key in [key for key, value in possible_value_.items() if
                                    key[1] == available_cell[0][1] and key not in available_cell]:
                            possible_value_[key] = [y for y in possible_value_[key] if y != x]
        return 0

    def solveSudoku(self):

        while True:
            checker = True
            # make copy of the current iteration of the puzzle

            puzzle_old = copy.deepcopy(self.sudokuPuzzle)

            self.loop_basic_rule(self.possible_value, self.sudokuPuzzle)
            self.loop_algorithm(self.possible_value, self.sudokuPuzzle)
            self.check_box_eliminate_others(self.possible_value)
            for i in range(1, 10):
                for j in range(1, 10):
                    if self.sudokuPuzzle[i - 1][j - 1] == puzzle_old[i - 1][j - 1]:
                        checker = False
            if not checker:
                break

        print("puzzle solved")
