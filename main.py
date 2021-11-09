import cv2 as cv
import numpy as np

#Process the Image to gray Color, and and return the Threshold Image
def processImage(image):
    grayImage = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blurImage = cv.GaussianBlur(grayImage, (5,5), cv.BORDER_DEFAULT)
    imgThreshold = cv.adaptiveThreshold(blurImage, 255, 1, 1, 11, 2)
    return imgThreshold

def biggestContour(contour):
    biggest = np.array([])
    max_area = 0
    count = 1

    for i in contour:
        count = count + 1
        area = cv.contourArea(i)
        print("Area = ", area)
        if area > 60:
            peri = cv.arcLength(i, True)
            approx = cv.approxPolyDP(i, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    print(count)
    return biggest, max_area


# Reorder the The biggest Contour cordinate points
def reorder(points):
    points = points.reshape((4, 2))
    newPoints = np.zeros((4, 1, 2), dtype=np.int32)
    add = points.sum(1)
    newPoints[0] = points[np.argmin(add)]
    newPoints[3] = points[np.argmax(add)]
    diff = np.diff(points, axis=1)
    newPoints[1] = points[np.argmin(diff)]
    newPoints[2] = points[np.argmax(diff)]
    return newPoints

# split the warp image for each digit box
def splitBoxes(img):
    rows = np.vsplit(img, 9)
    boxes = []
    for r in rows:
        cols = np.hsplit(r , 9)
        for box in cols:
            boxes.append(box)
    return boxes

img = cv.imread("sudoku.jpg")
imgHeight = 450
imgWidth = 450

img = cv.resize(img, (imgWidth, imgHeight))

imgThres = processImage(img)

# Find all the Contours
imgContours = img.copy()
imgBigContours = img.copy()
_, contours, _ = cv.findContours(imgThres, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
cv.drawContours(imgContours, contours, -1, (0, 255, 0), 3)

blankImg = np.zeros((imgHeight, imgWidth, 3), np.uint8 )

# Find the bigest contours, and warp it
biggest, maxArea = biggestContour(contours)
cv.drawContours(imgBigContours, biggest, -1, (0, 0, 255), 10)
print(biggest, "Max Area = ", maxArea)
if biggest.size != 0:
    biggest = reorder(biggest)
    print(biggest)
    cv.drawContours(imgBigContours, biggest, -1, (0, 0, 255), 3)
    pnts1 = np.float32(biggest)
    pnts2 = np.float32([[0,0], [imgWidth, 0], [0, imgHeight], [imgWidth, imgHeight]])
    matrix = cv.getPerspectiveTransform(pnts1, pnts2)
    imgWarpColored = cv.warpPerspective(img, matrix, (imgWidth, imgHeight))
    imgDetectedDigits = blankImg.copy()
    imgWarpColored = cv.cvtColor(imgWarpColored, cv.COLOR_BGR2GRAY)

# Split The Digit
boxes = splitBoxes(imgWarpColored)
print("Boxes length",len(boxes))

cv.imshow("Sudoko Image", img)
cv.imshow("Threshold Image", imgThres)
cv.imshow("All Image Contours", imgContours)
cv.imshow("biggest Countour Points", imgBigContours)
cv.imshow("Image warp", imgWarpColored)
cv.imshow("boxes at 2", boxes[11])
cv.waitKey(0)