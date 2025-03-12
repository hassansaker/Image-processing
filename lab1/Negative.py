import cv2 as cv
import matplotlib.pyplot as plt 
import functions


def main():

    img=functions.read_image('breast.jpg')

    plt.figure('photo')
    plt.subplot(2,2,1)
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis('off')  # Turn off axis labels

    im_gray=cv.cvtColor(img,cv.COLOR_RGB2GRAY)
    negative_image1 = functions.apply_negative_transform_1(img)
    negative_image2 = functions.apply_negative_transform_2(img)
    
    plt.subplot(2, 2, 2)  # Subplot for the negative image
    plt.imshow(cv.cvtColor(im_gray, cv.COLOR_BGR2RGB))
    plt.title("gray scale Image")
    plt.axis('off')

    plt.subplot(2, 2, 3)  # Subplot for the negative image
    plt.imshow(cv.cvtColor(negative_image1, cv.COLOR_BGR2RGB))
    plt.title("Negative Image 1")
    plt.axis('off')

    plt.subplot(2, 2, 4)  # Subplot for the negative image
    plt.imshow(cv.cvtColor(negative_image2, cv.COLOR_BGR2RGB))
    plt.title("Negative Image 2")
    plt.axis('off')
    
    functions.plot_Histogram2(im_gray,negative_image1,'GrayScale Image','Negative Image')

    plt.show()



main()