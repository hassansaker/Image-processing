import cv2 as cv
import matplotlib.pyplot as plt 
import functions


def main():

    img=functions.read_image('flower.jpg')
    im_gray=cv.cvtColor(img,cv.COLOR_RGB2GRAY)
    plt.figure('photo')
    plt.subplot(1,2,1)
    plt.imshow(cv.cvtColor(im_gray, cv.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis('off')  # Turn off axis labels

    im_eq=functions.histogram_equalizer(im_gray)
    plt.figure('photo')
    plt.subplot(1,2,2)
    plt.imshow(cv.cvtColor(im_eq, cv.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis('off')  # Turn off axis labels

    functions.plot_Histogram2(im_gray,im_eq,'GrayScale Image','Equalized Image')
    plt.show()

main()