# Image Processing Toolkit

This project provides a collection of Python scripts for performing various image processing tasks, including grayscale conversion, negative transformation, histogram equalization, gamma correction, logarithmic transformation, and more. The project uses OpenCV, NumPy, and Matplotlib for image manipulation and visualization.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
  - [Negative.py](#negativepy)
  - [Equalizer.py](#equalizerpy)
  - [gamma.py](#gammapy)
  - [Logarithm.py](#logarithmpy)
  - [constract.py](#constractpy)
- [Functions](#functions)
- [License](#license)

## Introduction

This project is designed to demonstrate various image processing techniques using Python. It includes the following scripts:
- **Negative.py**: Applies negative transformations to an image.
- **Equalizer.py**: Performs histogram equalization to enhance image contrast.
- **gamma.py**: Applies gamma correction to adjust image brightness.
- **Logarithm.py**: Applies logarithmic transformation to an image.
- **constract.py**: Demonstrates histogram stretching for contrast enhancement.

The project also includes a `functions.py` module that contains utility functions for reading images, converting them to grayscale, applying transformations, and plotting results.

## Installation

To use this project, you need to have Python installed along with the following libraries:
- `numpy`
- `opencv-python`
- `matplotlib`

You can install these libraries using pip:

```bash
pip install numpy opencv-python matplotlib
