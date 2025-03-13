import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import functions

def main():
    # Read the image and convert it to grayscale
    img = functions.read_image('flower.JPG')
    im_gray = functions.transform_to_gray(img)

    # Define the sharpening kernel
    sharpening_kernel = np.array([[0, -1, 0],
                                  [-1, 5, -1],
                                  [0, -1, 0]])

    # Apply the sharpening filter using cv2.filter2D
    sharpened_img = cv.filter2D(im_gray, -1, sharpening_kernel)
    # Display the original grayscale image and the sharpened image
    functions.plot2img(im_gray,sharpened_img,'image','sharpened image','Sharpening Filter')

    plt.show()


main()
