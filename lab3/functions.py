import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

def read_image(path):
    
    # Reads an image from the specified file path.
    # Parameters:
    # path (str): The file path to the image.
    # Returns:
    # numpy.ndarray: The image read from the file, or None if the image could not be loaded.
    
    return cv.imread(path)

def transform_to_gray(image):

    # Converts a color image to grayscale.
    # Parameters:
    # image (numpy.ndarray): The input color image in RGB format.
    # Returns:
    # numpy.ndarray: The grayscale version of the input image.

    return cv.cvtColor(image, cv.COLOR_RGB2GRAY)

def apply_negative_transform_1(img):

    # Applies a negative transformation to the image using pixel inversion.
    # Parameters:
    # img (numpy.ndarray): The input image.
    # Returns:
    # numpy.ndarray: The negative of the input image.

    return 255 - img

def apply_negative_transform_2(img):

    # Applies a negative transformation to the image using bitwise NOT operation.
    # Parameters:
    # img (numpy.ndarray): The input image.
    # Returns:
    # numpy.ndarray: The negative of the input image.

    return cv.bitwise_not(img)

def plot_Histogram(image, title):

    # Plots the histogram of a single grayscale image.
    # Parameters:
    # image (numpy.ndarray): The input grayscale image.
    # title (str): The title for the histogram plot.
    # This function creates a histogram representing the distribution of pixel intensities in the image.

    plt.figure('Histogram')
    
    # Calculate histogram for the grayscale image
    hist_original = cv.calcHist([image], [0], None, [256], [0, 256])
    
    # Plot the histogram
    plt.plot(hist_original)
    
    # Set title for the histogram plot
    plt.title(title)

def plot_Histogram2(image1, image2, title1='', title2=''):

    # Plots histograms for two images side by side for comparison.
    # Parameters:
    # image1 (numpy.ndarray): The first input grayscale image.
    # image2 (numpy.ndarray): The second input grayscale image.
    # title1 (str): Title for the first histogram plot.
    # title2 (str): Title for the second histogram plot.
    # This function allows visual comparison of pixel intensity distributions between two images.

    plt.figure('Histogram')

    # Calculate histograms for both images
    hist_1 = cv.calcHist([image1], [0], None, [256], [0, 256])
    hist_2 = cv.calcHist([image2], [0], None, [256], [0, 256])
    
    # Plot first histogram
    plt.subplot(1, 2, 1)
    plt.plot(hist_1)
    
    # Set title for first histogram plot
    plt.title(title1)
    
    # Plot second histogram
    plt.subplot(1, 2, 2)
    plt.plot(hist_2)
    
    # Set title for second histogram plot
    plt.title(title2)

def plot_Histogram3(image1, image2,image3, title1='', title2='',title3=''):

    # Plots histograms for two images side by side for comparison.
    # Parameters:
    # image1 (numpy.ndarray): The first input grayscale image.
    # image2 (numpy.ndarray): The second input grayscale image.
    # image3 (numpy.ndarray): The third input grayscale image.
    # title1 (str): Title for the first histogram plot.
    # title2 (str): Title for the second histogram plot.
    # title3 (str): Title for the third histogram plot.
    # This function allows visual comparison of pixel intensity distributions between two images.

    plt.figure('Histogram')

    # Calculate histograms for both images
    hist_1 = cv.calcHist([image1], [0], None, [256], [0, 256])
    hist_2 = cv.calcHist([image2], [0], None, [256], [0, 256])
    hist_3 = cv.calcHist([image3], [0], None, [256], [0, 256])
    # Plot first histogram
    plt.subplot(1, 3, 1)
    plt.plot(hist_1)
    
    # Set title for first histogram plot
    plt.title(title1)
    
    # Plot second histogram
    plt.subplot(1, 3, 2)
    plt.plot(hist_2)

    # Set title for second histogram plot
    plt.title(title2)

     # Plot third histogram
    plt.subplot(1, 3, 3)
    plt.plot(hist_3)

    # Set title for third histogram plot
    plt.title(title3)

def histogram_stretching(img):
   
    # Applies histogram stretching to enhance contrast in an image.
    # Parameters:
    # img (numpy.ndarray): The input grayscale image.
    # Returns:
    # numpy.ndarray: The contrast-enhanced image after histogram stretching.
    # This function normalizes pixel values to span the full range of [0, 255].
    # It improves visibility in images with low contrast by redistributing pixel intensitie
   
   # Get minimum and maximum pixel values in the image
   min_val = img.min()
   max_val = img.max()
   
   # Apply histogram stretching formula
   stretched_img = (img - min_val) * (255 / (max_val - min_val))
   
   # Ensure pixel values are within valid range [0, 255]
   stretched_img = np.clip(stretched_img, 0, 255).astype(np.uint8)
   
   return stretched_img

def gamma_transformation(image, gamma):

    #  Applies gamma correction to an image to adjust brightness and contrast.
    #  Parameters:
    #  image (numpy.ndarray): The input grayscale or color image.
    #  gamma (float): The gamma value used for correction. Values > 1 darken the image; values < 1 brighten it.
    #  Returns:
    #  numpy.ndarray: The gamma-corrected output image.
    #  This function normalizes pixel values to a range of [0, 1], applies gamma correction,
    #  and scales back to the original range of [0, 255].

     
     # Normalize the input image to range [0, 1]
     normalized = image / 255.0
     
     # Apply gamma correction using the specified gamma value
     gamma_corrected = np.power(normalized, gamma)
     
     # Scale back to range [0, 255]
     output = (gamma_corrected * 255).astype(np.uint8)
     
     return output

def log_transform(image):

    # Apply logarithmic transformation to an image
    # Parameters:
    # image (numpy.ndarray): Input image in BGR format
    # Returns:
    # numpy.ndarray: Log-transformed image

    # Check if the image is valid
    if image is None:
        raise ValueError("Image not found or invalid.")

    # Convert to float32 to avoid overflow during calculations
    image_float = np.float32(image)

    # Apply log transformation
    c = 255 / np.log(1 + np.max(image_float))
    log_image = c * (np.log(image_float + 1))

    # Convert back to uint8
    log_image = np.array(log_image, dtype=np.uint8)
    
    return log_image

def histogram_equalizer(image):

    # Apply Histogram Equalization to an image
    # Parameters:
    # image (numpy.ndarray): Input image in BGR format
    # Returns:
    # numpy.ndarray: Equilized image

    # Check if the image is valid
    if image is None:
        raise ValueError("Image not found or invalid.")
    # Apply histogram equalization
    equalized_image = cv.equalizeHist(image)

    return equalized_image

def plot2img(img1,img2,title1,title2,title):
    plt.figure(title)
    plt.subplot(1, 2, 1)  
    plt.imshow(img1,cmap='gray')
    plt.title(title1)
    plt.axis('off')

    plt.subplot(1, 2, 2)  
    plt.imshow(img2,cmap='gray')
    plt.title(title2)
    plt.axis('off')
    plt.tight_layout()

def plot4img(img1,img2,img3,img4,title1,title2,title3,title4,title):
    plt.figure(title)
    plt.subplot(2, 2, 1)  
    plt.imshow(img1, cmap='gray')
    plt.title(title1)
    plt.axis('off')

    plt.subplot(2, 2, 2)  
    plt.imshow(img2, cmap='gray')
    plt.title(title2)
    plt.axis('off')

    plt.subplot(2, 2, 3)  
    plt.imshow(img3, cmap='gray')
    plt.title(title3)
    plt.axis('off')

    plt.subplot(2, 2, 4)  
    plt.imshow(img4, cmap='gray')
    plt.title(title4)
    plt.axis('off')        
    plt.tight_layout()

def sobel_edge_detection(image,kerneksize):

    # Calculate Sobel derivatives in the x and y directions
    sobel_x = cv.Sobel(image, cv.CV_64F, 1, 0, ksize=kerneksize)  # X direction
    sobel_y = cv.Sobel(image, cv.CV_64F, 0, 1, ksize=kerneksize)  # Y direction

    return sobel_x,sobel_y

def canny_edge_detection(image,low_threshold,high_threshold):

    edges = cv.Canny(image, low_threshold, high_threshold)
    plot2img(image,edges,'Grayscale','edge detection','canny Method')
    return edges
    
def laplacian_edge_detection(image):

    # Apply Laplacian of Gaussian (LoG) edge detector
    laplacian = cv.Laplacian(image, cv.CV_64F)
    laplacian = cv.convertScaleAbs(laplacian)
    plot2img(image,laplacian,'Grayscale','edge detection','LoG Method')
    return laplacian

#  Transform the image to the frequency domain using Fourier transform
def apply_fourier_transform(image):
    f_transform = np.fft.fft2(image)  # Compute the 2D Fourier Transform
    f_transform_shifted = np.fft.fftshift(f_transform)  # Shift zero frequency to the center
    return f_transform_shifted

# print the details of an image
def imagDetails(image):
    print(f"Max Value: {np.max(image)}")
    print(f"Min Value: {np.min(image)}")
    print(f"Image Type: {image.dtype}")

# Return the image to the spatial domain using inverse Fourier transform
def apply_inverse_fourier_transform(f_transform_shifted):
    f_ishift = np.fft.ifftshift(f_transform_shifted)  # Inverse shift
    img_reconstructed = np.fft.ifft2(f_ishift)  # Inverse Fourier Transform
    img_reconstructed = np.abs(img_reconstructed)  # Get real values
    return img_reconstructed

# Create a function to print and apply details in the frequency domain
def imShowFreqDomain(f_transform_shifted):
    magnitude_spectrum = np.abs(f_transform_shifted)
    c = 255 / np.log(1 + np.max(magnitude_spectrum))  # Calculate the scaling constant
    log_transformed = c * np.log(1 + magnitude_spectrum)  # Apply log transformation
    log_transformed = np.uint8(log_transformed)  # Convert to uint8 for display
    plt.figure(figsize=(6, 6))
    plt.title("Log Transformed Frequency Domain")
    plt.imshow(log_transformed, cmap='gray')
    plt.axis('off')

# Create a low-pass filter mask
def create_low_pass_filter(image_shape, cutoff_frequency):
    rows, cols = image_shape
    center_row, center_col = rows // 2, cols // 2
    mask = np.zeros((rows, cols), dtype=np.uint8)
    radius = int(cutoff_frequency * min(rows, cols) / 2)  # Cutoff based on image size
    cv.circle(mask, (center_col, center_row), radius, 1, -1)
    return mask

# Create a high-pass filter mask
def create_high_pass_filter(image_shape, cutoff_frequency):
    rows, cols = image_shape
    center_row, center_col = rows // 2, cols // 2
    mask = np.ones((rows, cols), dtype=np.uint8)
    radius = int(cutoff_frequency * min(rows, cols) / 2)  # Cutoff based on image size
    cv.circle(mask, (center_col, center_row), radius, 0, -1)
    return mask

# Apply the low-pass filter in the frequency domain
def apply_filter_in_freq(f_transform_shifted, filter_mask):
    filtered_transform = f_transform_shifted * filter_mask
    return filtered_transform