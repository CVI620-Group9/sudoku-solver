import cv2 as cv
import numpy as np

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

model = load_model('../myModel.h5')

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