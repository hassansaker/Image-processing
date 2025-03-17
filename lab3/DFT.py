import numpy as np
import cv2
import matplotlib.pyplot as plt
import functions



def main():
    
    image_path = 'text.jpg'  # Replace with your image path
    img=functions.read_image(image_path)
    # transform image to grayscale
    im_gray=functions.transform_to_gray(img)
    # Apply Fourier Transform
    f_transform_shifted = functions.apply_fourier_transform(im_gray)
    magnitude_spectrum = 20 * np.log(np.abs(f_transform_shifted) + 1)
    # Display the image and Fourier Transform
    functions.plot2img(img,magnitude_spectrum,'Image','Fourier Trasnform','spatial domain vs frequecny domain')

    # Display image details
    functions.imagDetails(img)

    # Show log-transformed frequency domain
    functions.imShowFreqDomain(f_transform_shifted)

    # Apply inverse Fourier Transform
    reconstructed_image = functions.apply_inverse_fourier_transform(f_transform_shifted)
    # Display the image and reconstructed image
    functions.plot2img(img,reconstructed_image,'Image','reconstructed image','reconstruction')
    
 
    plt.show()

main()