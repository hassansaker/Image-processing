import cv2 as cv
import matplotlib.pyplot as plt   
import numpy as np   
import skimage.util as ski
import functions

def main():

    img=functions.read_image('edgeflower.jpg')
    im_gray=functions.transform_to_gray(img)
    noissy_img=ski.random_noise(im_gray, mode='gaussian', mean=0, var=0.1)
    noissy_img=(255 * noissy_img).astype(np.uint8)
    functions.plot2img(im_gray,noissy_img,'Grayscale','Gaussian noise','the image')
    # ############################################################################################
    # # Sobel Method
    # sobel_x, sobel_y = functions.sobel_edge_detection(image=im_gray, kerneksize=3)
    # sobel_x_n, sobel_y_n = functions.sobel_edge_detection(image=noissy_img, kerneksize=3)
    # gradient = np.sqrt(np.power(sobel_x, 2) + np.power(sobel_y, 2))
    # gradient_n = np.sqrt(np.power(sobel_x_n, 2) + np.power(sobel_y, 2))
    # gradient = cv.convertScaleAbs(gradient)
    # gradient_n = cv.convertScaleAbs(gradient_n)
    # _, binary_edge = cv.threshold(gradient, 50, 255, cv.THRESH_BINARY)
    # _, binary_edge_n = cv.threshold(gradient_n,200, 255, cv.THRESH_BINARY)
    # functions.plot4img(im_gray,noissy_img,binary_edge,binary_edge_n,'Grayscale','noissy image','edge detection for image',
    #                    'edge detection for noissy image','Sobel Method')

    ############################################################################################
    # # Canny Method
    # low_threshold, high_threshold = 50, 100
    # edges=functions.canny_edge_detection(im_gray, low_threshold, high_threshold)
    # # Apply Gaussian blur to reduce noise
    # blurred_image = cv.GaussianBlur(noissy_img, (15,15),0)
    # edges_n = cv.Canny(blurred_image, low_threshold, high_threshold)
    # functions.plot4img(im_gray,noissy_img,edges,edges_n,'Grayscale','edge detection',
    #                    'edge detection for image','edge detection for noissy image','canny Method')

    ############################################################################################
    # # # LoG Method
    edges=functions.laplacian_edge_detection(im_gray)
    blurred_image = cv.GaussianBlur(noissy_img, (3,3),0)
    edges_n = functions.laplacian_edge_detection(blurred_image)
    functions.plot4img(im_gray,noissy_img,edges,edges_n,'Grayscale','edge detection',
                       'edge detection for image','edge detection for noissy image','LoG Method')    

    plt.show()



main()