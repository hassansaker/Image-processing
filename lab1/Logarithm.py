import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import functions



def main():

    img=functions.read_image('flowers.jpg')

    im_gray=cv.cvtColor(img,cv.COLOR_RGB2GRAY)
    plt.figure('photo')
    plt.subplot(1,2,1)
    plt.imshow(cv.cvtColor(im_gray, cv.COLOR_BGR2RGB))
    plt.title("GrayScale Image")
    plt.axis('off')  # Turn off axis labels

    im_log=functions.log_transform(im_gray)
    plt.subplot(1, 2, 2)  # Subplot for the negative image
    plt.imshow(cv.cvtColor(im_log, cv.COLOR_BGR2RGB))
    plt.title("logarithm transformation")
    plt.axis('off')
    
    functions.plot_Histogram2(im_gray,im_log,'GrayScale Image','logarithm transformation')
    
    
    plt.show()



main()