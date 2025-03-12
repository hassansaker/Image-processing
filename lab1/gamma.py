import cv2 as cv
import matplotlib.pyplot as plt 
import numpy as np
import functions



def main():

    img=functions.read_image('raider.png')

    plt.figure('photo')
    plt.subplot(2,2,1)
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis('off')  # Turn off axis labels

    im_gray=cv.cvtColor(img,cv.COLOR_RGB2GRAY)
    plt.subplot(2, 2, 2)  # Subplot for the negative image
    plt.imshow(cv.cvtColor(im_gray, cv.COLOR_BGR2RGB))
    plt.title("gray scale Image")
    plt.axis('off')

    im_gamma1=functions.gamma_transformation(im_gray,0.4)
    plt.subplot(2, 2, 3)  # Subplot for the negative image
    plt.imshow(cv.cvtColor(im_gamma1, cv.COLOR_BGR2RGB))
    plt.title("gamma=0.4")
    plt.axis('off')

    im_gamma2=functions.gamma_transformation(im_gray,0.7)
    plt.subplot(2, 2, 4)  # Subplot for the negative image
    plt.imshow(cv.cvtColor(im_gamma2, cv.COLOR_BGR2RGB))
    plt.title("gamma=0.7")
    plt.axis('off')
    
    functions.plot_Histogram3(im_gray,im_gamma1,im_gamma2,'GrayScale Image','gamma= 0.4','gamma= 0.7')
    #functions.plot_Histogram2(im_gray,im_gamma2,'GrayScale Image','gammetransformation 0.7')

    plt.show()



main()