import cv2 as cv
import numpy as np
import os
import sys
from modules.SudokuManager import SudokuManager

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import modules.utils as utils

if __name__ == "__main__":
    img = cv.imread(sys.argv[1])
    imgHeight = 450
    imgWidth = 450

    img = cv.resize(img, (imgWidth, imgHeight))

    imgThres = utils.processImage(img)

    # Find all the Contours
    imgContours = img.copy()
    imgBigContours = img.copy()
    contours, _ = cv.findContours(imgThres, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(imgContours, contours, -1, (0, 255, 0), 3)

    blankImg = np.zeros((imgHeight, imgWidth, 3), np.uint8)

    # Find the bigest contours, and warp it
    biggest, maxArea = utils.biggestContour(contours)
    cv.drawContours(imgBigContours, biggest, -1, (0, 0, 255), 10)
    print(f"Biggest: {biggest}")

    if biggest.size != 0:
        biggest = utils.reorder(biggest)
        cv.drawContours(imgBigContours, biggest, -1, (0, 0, 255), 3)
        pnts1 = np.float32(biggest)
        pnts2 = np.float32([[0, 0], [imgWidth, 0], [0, imgHeight], [imgWidth, imgHeight]])
        matrix = cv.getPerspectiveTransform(pnts1 + 1, pnts2)
        imgWarpColored = cv.warpPerspective(img, matrix, (imgWidth, imgHeight))
        imgDetectedDigits = blankImg.copy()
        

    # Create sudoku manager
    sudo = SudokuManager(imgWarpColored)
    sudo.solveSudoku()
    sudo.printSudoku()

    cv.imshow("Result", sudo.sudokuImage)
    # wait 10 seconds
    cv.waitKey(10000)

    # Specifically destroy the window of the new image
    cv.destroyWindow("Result")
