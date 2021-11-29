import cv2 as cv
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import modules.utils as utils


if __name__ == "__main__":
    img = cv.imread("sudoku.jpg")
    imgHeight = 450
    imgWidth = 450

    img = cv.resize(img, (imgWidth, imgHeight))

    imgThres = utils.processImage(img)

    # Find all the Contours
    imgContours = img.copy()
    imgBigContours = img.copy()
    contours, _ = cv.findContours(imgThres, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(imgContours, contours, -1, (0, 255, 0), 3)

    blankImg = np.zeros((imgHeight, imgWidth, 3), np.uint8 )

    # Find the bigest contours, and warp it
    biggest, maxArea = utils.biggestContour(contours)
    cv.drawContours(imgBigContours, biggest, -1, (0, 0, 255), 10)

    if biggest.size != 0:
        biggest = utils.reorder(biggest)
        cv.drawContours(imgBigContours, biggest, -1, (0, 0, 255), 3)
        pnts1 = np.float32(biggest)
        pnts2 = np.float32([[0,0], [imgWidth, 0], [0, imgHeight], [imgWidth, imgHeight]])
        matrix = cv.getPerspectiveTransform(pnts1 + 1, pnts2)
        imgWarpColored = cv.warpPerspective(img, matrix, (imgWidth, imgHeight))
        imgDetectedDigits = blankImg.copy()
        imgWarpColored = cv.cvtColor(imgWarpColored, cv.COLOR_BGR2GRAY)

    # Split The Digit
    #boxes = utils.splitBoxes(imgWarpColored)

    #numbers = utils.getPredection(boxes, utils.model)
    #print(numbers)


    #cv.imshow("Sudoko Image", img)
    #cv.imshow("Threshold Image", imgThres)
    #cv.imshow("All Image Contours", imgContours)
    #cv.imshow("biggest Countour Points", imgBigContours)
    #cv.imshow("Image warp", imgWarpColored)
    #cv.imshow("boxes", boxes[0])
    #cv.waitKey(2000)

    cv.imshow("Sudoko Image", img)
    cv.imshow("Image warp", imgWarpColored)
    cv.waitKey(2000)




