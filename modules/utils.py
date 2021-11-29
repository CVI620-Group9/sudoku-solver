import cv2 as cv
import imutils
import numpy as np
import os

from modules.DrawInfo import DrawInfo
from tensorflow.keras.models import load_model

def showAndSave(image, imageName, isSaving):
    '''
    Saves the image to file as a jpg;
    Displays the image in a seperate window

    Parameters: 
        image: the image that is to be saved and displayed
        isSaving: boolean inidicating whether user wants the image to be saved
        saveName: the name that the image will be saved as
    
    Returns:
        Nothing
    '''
    if isSaving:
        # Save the modified image to disk
        cv.imwrite(f'{imageName}.jpg', image)

    # Show image
    cv.imshow(f"{imageName}", image)

    # wait 1 second
    cv.waitKey(5000)

    # Specifically destroy the window of the new image
    cv.destroyWindow(f"{imageName}")

#model = load_model('D:\CVI620\ProjectStuffs\Sudoku\sudoku-solver\myModel.h5')
modelPath = os.path.abspath('myModel.h5')
model = load_model(modelPath)

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
        if area > 60:
            peri = cv.arcLength(i, True)
            approx = cv.approxPolyDP(i, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    #print(count)
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


def getPredection(boxes,model):
    output = []
    for image in boxes:
        ## PREPARE IMAGE
        img = np.asarray(image)
        img = img[4:img.shape[0] - 4, 4:img.shape[1] -4]
        img = cv.resize(img, (28, 28))
        img = img / 255
        img = img.reshape(1, 28, 28, 1)
        ## GET PREDICTION
        predictions = model.predict(img)
        classIndex = model.predict_classes(img)
        predictedValue = np.amax(predictions)
        ## SAVE TO RESULT
        if predictedValue > 0.7:
            output.append(classIndex[0])
        else:
            output.append(0)
    return output



def drawRectangle(drawInfoObject, image, colour):
        if isinstance(drawInfoObject, DrawInfo):
            cv.rectangle(image, (drawInfoObject.x1, drawInfoObject.y1), (drawInfoObject.x2, drawInfoObject.y2), colour, 2)


from enum import Enum
# Colour Enum
class Colour(Enum):
    RED = (0,0,255)
    WHITE = (255,255,255)
    GREEN = (0,255,0)
    YELLOW = (0,255,255)
    FUSHIA = (255, 0, 255)


def createNumberAssets(directoryOfImages, savePath):
    # Loop through the files of a dir
    dirList = os.listdir(directoryOfImages)
    print(dirList)
    # Load the image
    for dir in dirList:
        img = cv.imread(directoryOfImages + "/" + dir)
        img_grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        thresh = cv.threshold(img_grey, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]

        cnts = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        digitCnts = []
        # loop over the digit area candidates
        for c in cnts:
	        # compute the bounding box of the contour
            x, y, w, h = cv.boundingRect(c)
            

	        # if the contour is sufficiently large, it must be a digit
            if w >= 15 and (h >= 25 and h <= 45):
                digitCnts.append(c)

        
        for d in digitCnts:
            # extract the digit ROI
            (x, y, w, h) = cv.boundingRect(d)
            
            # Extract image from original and save
            imgToSave = img[y:y+h, x:x+w]
            # Save image
            cv.imwrite(savePath + "/" + dir, imgToSave)


def getNumberImage(num, assetPath):
    # Assumption that the file is a jpg
    numberImg = cv.imread(assetPath + f"/{num}.JPG")

    return numberImg

def changeTextColour(img, colour):
    # Load the aerial image and convert to HSV colourspace
    hsv = cv.cvtColor(img,cv.COLOR_BGR2HSV)

    # Define lower and uppper limits of what we call "black"
    black_low = np.array([0,0,0])
    black_high =np.array([190,190,190])

    # Mask image to only select black
    mask = cv.inRange(hsv, black_low, black_high)

    # Change image to colour where we found black
    img[mask>0] = colour

    return img