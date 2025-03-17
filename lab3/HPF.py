import numpy as np
import cv2
import matplotlib.pyplot as plt
import functions

def main():
    
    path = 'football.jpg'  # Replace with the path to your image
    imag=functions.read_image(path)
    # transform image to grayscale
    imag_gray = functions.transform_to_gray(imag)

    # Transform to frequency domain
    f_transform_shifted = functions.apply_fourier_transform(imag_gray)
    magnitude_spectrum = 20 * np.log(np.abs(f_transform_shifted) + 1)
    # Create and apply low-pass filter
    cutoff_frequency1 = 0.15  # first cutoff frequency
    cutoff_frequency2 = 0.3  # second cutoff frequency
    filter_mask_1 = functions.create_high_pass_filter(imag_gray.shape, cutoff_frequency1)
    filter_mask_2 = functions.create_high_pass_filter(imag_gray.shape, cutoff_frequency2)
    filtered_transform1 = functions.apply_filter_in_freq(f_transform_shifted, filter_mask_1)
    filtered_transform2 = functions.apply_filter_in_freq(f_transform_shifted, filter_mask_2)
    # Reconstruct the image
    reconstructed_image1 = functions.apply_inverse_fourier_transform(filtered_transform1)
    reconstructed_image2 = functions.apply_inverse_fourier_transform(filtered_transform2)
    # Display results
    functions.plot4img(imag_gray,magnitude_spectrum,filter_mask_1,reconstructed_image1,'Image',
                       'Fourier transform','filter Mask','reconstructed image','cuttoff =0.15'
                       )
    functions.plot4img(imag_gray,magnitude_spectrum,filter_mask_2,reconstructed_image2,'Image',
                       'Fourier transform','filter Mask','reconstructed image','cuttoff =0.3'
                       )
    plt.show()

main()
