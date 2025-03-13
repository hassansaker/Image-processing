import cv2 as cv
import matplotlib.pyplot as plt   
import numpy as np   
import skimage.util as ski
import functions

def main():

    
    # Read the original image and convert it to grayscale
    img = functions.read_image('nature.jpg')
    im_gray = functions.transform_to_gray(img)

    # Add Gaussian noise
    gauss_noise_img = ski.random_noise(im_gray, mode='gaussian', mean=0, var=0.1)
    gauss_noise_img = (255 * gauss_noise_img).astype(np.uint8) 

    # Add Salt & Pepper noise
    sp_noise_img = ski.random_noise(im_gray, mode='s&p', amount=0.25)
    sp_noise_img = (255 * sp_noise_img).astype(np.uint8)  

    ######################################################################################
    ## Average Filter 
    # Apply Average Filter 5x5
    avg_gauss1 = cv.blur(gauss_noise_img, (5,5))
    avg_sp1 = cv.blur(sp_noise_img, (5, 5))

    functions.plot4img(img1=gauss_noise_img,img2=sp_noise_img,img3=avg_gauss1,img4=avg_sp1,title1='Gaussian noise',
     title2='Salt & Pepper noise',title3='Average Filter 5x5',title4='Average Filter 5x5',title='Average Filter 5x5 for two types of Noise'                  
    )
    # try different Filter with different dimension
    # Apply Average Filter 3x3
    avg_gauss2 = cv.blur(gauss_noise_img, (3,3))
    avg_sp2 = cv.blur(sp_noise_img, (3,3))

    # Apply Average Filter 11x11
    avg_gauss3 = cv.blur(gauss_noise_img, (11, 11))
    avg_sp3 = cv.blur(sp_noise_img, (11, 11))

    # Apply Average Filter 21x21
    avg_gauss4 = cv.blur(gauss_noise_img, (21, 21))
    avg_sp4 = cv.blur(sp_noise_img, (21, 21))

    functions.plot4img(img1=avg_gauss2,img2=avg_gauss1,img3=avg_gauss3,img4=avg_gauss4,title1='3x3 Filter',
     title2='5x5 Filter',title3='11x11 Filter',title4='21x21 Filter',title='Different Size Average Filter'                  
    )

    # ######################################################################################
    # ## Apply Median Filter
    # # Apply Median Filter 5x5
    median_gauss1 = cv.medianBlur(gauss_noise_img, 5)
    median_sp1 = cv.medianBlur(sp_noise_img, 5)

    functions.plot4img(img1=gauss_noise_img,img2=sp_noise_img,img3=median_gauss1,img4=median_sp1,title1='Gaussian noise',
     title2='Salt & Pepper noise',title3='Median Filter 5x5',title4='Median Filter 5x5',title='Median Filter'                  
    )
    # try different Filter with different dimension
    # Apply Median Filter 3x3
    median_gauss2 = cv.medianBlur(gauss_noise_img, 3)
    median_sp2 = cv.medianBlur(sp_noise_img, 3)

    # Apply Median Filter 11x11
    median_gauss3 = cv.medianBlur(gauss_noise_img, 11)
    median_sp3 = cv.medianBlur(sp_noise_img, 11)

    # Apply Median Filter 21x21
    median_gauss4 = cv.medianBlur(gauss_noise_img, 21)
    median_sp4 = cv.medianBlur(sp_noise_img, 21)

    functions.plot4img(img1=median_sp2,img2=median_sp1,img3=median_sp3,img4=median_sp4,title1='3x3 Filter',
     title2='5x5 Filter',title3='11x11 Filter',title4='21x21 Filter',title='Different Size Median Filter'                  
    )
    # ######################################################################################
    # ## Apply Gaussian Filter
    # # Apply Gaussian Filter 5x5    
    gaussian_gauss1 = cv.GaussianBlur(gauss_noise_img, (5, 5), 0)
    gaussian_sp1 = cv.GaussianBlur(sp_noise_img, (5, 5), 0)
    # The last parameter is our sigma, the standard deviation of the Gaussian distribution. ...
    # By setting this value to 0, we are instructing OpenCV to automatically compute sigma based on...
    # our kernel size.
    functions.plot4img(img1=gauss_noise_img,img2=sp_noise_img,img3=gaussian_gauss1,img4=gaussian_sp1,title1='Gaussian noise',
     title2='Salt & Pepper noise',title3='Gaussian Filter 5x5',title4='Gaussian Filter 5x5',title='Gaussian Filter'                  
    )
    # try different Filter with different dimension
    # Apply Gaussian Filter 3x3
    gaussian_gauss2 = cv.GaussianBlur(gauss_noise_img, (3, 3), 0)
    gaussian_sp2 = cv.GaussianBlur(sp_noise_img, (3, 3), 0)

    # Apply Gaussian Filter 11x11
    gaussian_gauss3 = cv.GaussianBlur(gauss_noise_img, (11, 11), 0)
    gaussian_sp3 = cv.GaussianBlur(sp_noise_img, (11, 11), 0)

    # Apply Gaussian Filter 21x21
    gaussian_gauss4 = cv.GaussianBlur(gauss_noise_img, (21, 21), 0)
    gaussian_sp4 = cv.GaussianBlur(sp_noise_img, (21, 21), 0)

    functions.plot4img(img1=gaussian_sp2,img2=gaussian_sp1,img3=gaussian_sp3,img4=gaussian_sp4,title1='3x3 Filter',
     title2='5x5 Filter',title3='11x11 Filter',title4='21x21 Filter',title='Different Size Gaussian Filter'                  
    )

    # ######################################################################################
    # ## Apply Bilateral Filter
    # # Apply Bilateral Filter 5x5    
    bilateral_gauss1 = cv.bilateralFilter(gauss_noise_img, 5, 75, 75)
    bilateral_sp1 = cv.bilateralFilter(sp_noise_img, 5, 75, 75)

    functions.plot4img(img1=gauss_noise_img,img2=sp_noise_img,img3=bilateral_gauss1,img4=bilateral_sp1,title1='Gaussian noise',
     title2='Salt & Pepper noise',title3='Bilateral Filter 5x5',title4='Bilateral Filter 5x5',title='Bilateral Filter'                  
    )
    # try different Filter with different dimension
    # Apply Gaussian Filter 3x3
    bilateral_gauss2 = cv.bilateralFilter(gauss_noise_img, 3, 75, 75)
    bilateral_sp2 = cv.bilateralFilter(sp_noise_img, 3, 75, 75)

    # Apply Gaussian Filter 11x11
    bilateral_gauss3 = cv.bilateralFilter(gauss_noise_img, 11, 75, 75)
    bilateral_sp3 = cv.bilateralFilter(sp_noise_img, 11, 75, 75)

    # Apply Gaussian Filter 21x21
    bilateral_gauss4 = cv.bilateralFilter(gauss_noise_img, 21, 75, 75)
    bilateral_sp4 = cv.bilateralFilter(sp_noise_img, 21, 75, 75)

    functions.plot4img(img1=bilateral_sp2,img2=bilateral_sp1,img3=bilateral_sp3,img4=bilateral_sp4,title1='3x3 Filter',
     title2='5x5 Filter',title3='11x11 Filter',title4='21x21 Filter',title='Different Size Bilateral Filter'                  
    )

    plt.show()


main()
