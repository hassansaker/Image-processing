import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import functions


def main():

    img = functions.read_image('coffee.png')
    
    plt.figure(1)
    plt.subplot(2, 2, 1)
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis('off')  # Turn off axis labels
    
    im_gray =functions.transform_to_gray(img)
    
    plt.subplot(2, 2, 2)  # Subplot for the gray scale image
    plt.imshow(im_gray, cmap='gray')
    plt.title("Gray Scale Image")
    plt.axis('off')

    negative_image = functions.apply_negative_transform_1(im_gray)

    plt.subplot(2, 2, 3)  # Subplot for the negative image
    plt.imshow(negative_image, cmap='gray')
    plt.title("Negative Image")
    plt.axis('off')

    # Apply histogram stretching
    img_stretch = functions.histogram_stretching(im_gray)

    plt.subplot(2, 2, 4)  # Subplot for the stretched image
    plt.imshow(img_stretch, cmap='gray')
    plt.title("Stretched Image")
    plt.axis('off')
    
    # Plot histograms
    functions.plot_Histogram2(im_gray,img_stretch,'GrayScale Image','Stretched Image')
    
    plt.show()

main()