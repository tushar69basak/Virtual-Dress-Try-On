import cv2
import numpy as np

def remove_background(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Convert BGR image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to get binary mask of background
    _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

    # Invert the binary mask
    binary = cv2.bitwise_not(binary)

    # Apply the mask to the original image to remove the background
    result = cv2.bitwise_and(image, image, mask=binary)

    return result
