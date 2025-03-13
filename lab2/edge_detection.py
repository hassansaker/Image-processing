import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import functions

def main():
    
    # Load the image and convert it to grayscale
    img = functions.read_image('figure.webp')
    im_gray = functions.transform_to_gray(img)

    ####################################################################################
    ## Sobel Method for Edge Detection
    # Calculate Sobel derivatives in x and y directions
    sobel_x, sobel_y = functions.sobel_edge_detection(image=im_gray, kerneksize=3)

    # Display the Sobel x and y derivatives side by side
    sobel_combined = np.hstack((sobel_x, sobel_y))
    plt.figure('Sobel X and Y Derivatives')
    plt.imshow(sobel_combined, cmap='gray')
    plt.title("Sobel X and Y Derivatives")
    plt.axis('off')

    # Calculate gradient magnitude from Sobel x and y derivatives
    gradient = np.sqrt(np.power(sobel_x, 2) + np.power(sobel_y, 2))
    max_gradient_value = np.max(gradient)
    print(f"Max Gradient Value: {max_gradient_value}")

    # Normalize gradient to the range [0, 255] for display purposes
    gradient = cv.convertScaleAbs(gradient)

    # Display the gradient magnitude image
    plt.figure('Gradient Magnitude')
    plt.imshow(gradient, cmap='gray')
    plt.title("Gradient Magnitude")
    plt.axis('off')

    # Apply thresholding to the gradient image to obtain binary edge images
    thresholds = [50, 100, 150, 200]  # Example threshold values
    _, binary_edge_1 = cv.threshold(gradient, thresholds[0], 255, cv.THRESH_BINARY)
    _, binary_edge_2 = cv.threshold(gradient, thresholds[1], 255, cv.THRESH_BINARY)
    _, binary_edge_3 = cv.threshold(gradient, thresholds[2], 255, cv.THRESH_BINARY)
    _, binary_edge_4 = cv.threshold(gradient, thresholds[3], 255, cv.THRESH_BINARY)

    # Display binary edge images with different threshold values
    functions.plot4img(binary_edge_1, binary_edge_2, binary_edge_3, binary_edge_4,
                       'Threshold = 50', 'Threshold = 100', 'Threshold = 150', 'Threshold = 200',
                       'Binary Edge Detection with Different Thresholds')
    
    ####################################################################################
    ## Canny Method for Edge Detection
    # Apply the Canny edge detector with specified low and high thresholds
    low_threshold, high_threshold = 20, 120
    functions.canny_edge_detection(im_gray, low_threshold, high_threshold)
    
    ####################################################################################
    ## Laplacian (LoG) Method for Edge Detection
    # Apply the Laplacian of Gaussian (LoG) method for edge detection
    functions.laplacian_edge_detection(im_gray)

    # Show all plots
    plt.show()


main()
