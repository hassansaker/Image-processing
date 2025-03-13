import cv2 as cv
import matplotlib.pyplot as plt   
import numpy as np   
import functions
import skimage.util as ski

def main():


    # Read and display the original image
    img = functions.read_image('nature.jpg')

    # Transform to grayscale and display
    im_gray = functions.transform_to_gray(img)

    # Add Gaussian noise and display
    gauss_noise = ski.random_noise(im_gray, mode='gaussian', mean=0, var=0.1)
    gauss_noise = (255 * gauss_noise).astype(np.uint8)

    # Add Salt & Pepper noise
    sp_noise = ski.random_noise(im_gray, mode='s&p', amount=0.25)
    sp_noise = (255 * sp_noise).astype(np.uint8) 

    # Stack images horizontally
    combined_img = np.hstack((cv.cvtColor(img, cv.COLOR_BGR2RGB), cv.cvtColor(im_gray, cv.COLOR_GRAY2RGB), 
                              cv.cvtColor(gauss_noise, cv.COLOR_GRAY2RGB), 
                              cv.cvtColor(sp_noise, cv.COLOR_GRAY2RGB)))

    # Display images
    plt.figure(figsize=(15, 8))
    plt.imshow(combined_img,aspect=1.66)
    plt.title("Original,  Grayscale,  Gaussian Noise,  Salt & Pepper Noise")
    plt.axis('off')
    plt.show()


main()