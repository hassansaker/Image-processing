# Image Processing with Fourier Transform

This project demonstrates various image processing techniques using the Fourier Transform. It includes scripts for applying low-pass and high-pass filters, as well as other image transformations such as histogram equalization, gamma correction, and edge detection.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
  - [DFT.py](#dftpy)
  - [LPF.py](#lpfpy)
  - [HPF.py](#hpfpy)
- [Functions](#functions)
- [License](#license)

## Introduction

The project consists of several Python scripts that perform image processing tasks using the Fourier Transform. The main scripts are:
- **DFT.py**: Applies the Discrete Fourier Transform (DFT) to an image and reconstructs it using the inverse DFT.
- **LPF.py**: Applies a low-pass filter to an image in the frequency domain.
- **HPF.py**: Applies a high-pass filter to an image in the frequency domain.

The project also includes a `functions.py` module that contains utility functions for reading images, converting them to grayscale, applying various transformations, and plotting results.

## Installation

To use this project, you need to have Python installed along with the following libraries:
- `numpy`
- `opencv-python`
- `matplotlib`

You can install these libraries using pip:

```bash
pip install numpy opencv-python matplotlib
