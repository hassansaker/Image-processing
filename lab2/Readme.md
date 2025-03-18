# Image Processing Project

This project demonstrates various image processing techniques using Python and OpenCV. The project includes modules for blurring, sharpening, noise addition, and edge detection. Below is an overview of the functionalities provided by each module.

## Modules

### 1. Blurring Filters (`Blurring_Filters.py`)
- **Functionality**: Applies different types of blurring filters to an image, including Average, Median, Gaussian, and Bilateral filters.
- **Usage**: 
  - Reads an image and converts it to grayscale.
  - Adds Gaussian and Salt & Pepper noise to the image.
  - Applies various blurring filters with different kernel sizes.
  - Displays the results using `matplotlib`.

### 2. Sharpening Filters (`Sharpening_Filters.py`)
- **Functionality**: Applies a sharpening filter to an image using a predefined kernel.
- **Usage**:
  - Reads an image and converts it to grayscale.
  - Applies a sharpening kernel using `cv2.filter2D`.
  - Displays the original and sharpened images.

### 3. Noise Addition (`Noised_Images.py`)
- **Functionality**: Adds Gaussian and Salt & Pepper noise to an image.
- **Usage**:
  - Reads an image and converts it to grayscale.
  - Adds Gaussian and Salt & Pepper noise.
  - Displays the original, grayscale, and noisy images.

### 4. Edge Detection (`edge_detection.py`)
- **Functionality**: Detects edges in an image using Sobel, Canny, and Laplacian of Gaussian (LoG) methods.
- **Usage**:
  - Reads an image and converts it to grayscale.
  - Applies Sobel, Canny, and LoG edge detection methods.
  - Displays the edge-detected images with different threshold values.

### 5. Comparison of Edge Detection Methods (`compare.py`)
- **Functionality**: Compares edge detection methods on a noisy image.
- **Usage**:
  - Reads an image and converts it to grayscale.
  - Adds Gaussian noise to the image.
  - Applies Sobel, Canny, and LoG edge detection methods.
  - Displays the results for comparison.

## Helper Functions (`functions.py`)
- **Functionality**: Contains utility functions for reading images, converting to grayscale, applying transformations, and plotting images.
- **Key Functions**:
  - `read_image(path)`: Reads an image from the specified path.
  - `transform_to_gray(image)`: Converts an image to grayscale.
  - `plot2img(img1, img2, title1, title2, title)`: Plots two images side by side.
  - `plot4img(img1, img2, img3, img4, title1, title2, title3, title4, title)`: Plots four images in a 2x2 grid.
  - `sobel_edge_detection(image, kerneksize)`: Applies Sobel edge detection.
  - `canny_edge_detection(image, low_threshold, high_threshold)`: Applies Canny edge detection.
  - `laplacian_edge_detection(image)`: Applies Laplacian of Gaussian (LoG) edge detection.

## Usage

1. **Install Dependencies**:
   ```bash
   pip install opencv-python matplotlib numpy scikit-image
