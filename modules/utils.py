import cv2 as cv

from modules.DrawInfo import DrawInfo

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