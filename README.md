# Circle-Detection

This project is a Python-based image processing tool that use OpenCV  to analyze images. It performs detects circles inside  in the image.

## Table of Contents

1. [Introduction](#introduction)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)


## Introduction

This project demonstrates how to use Python libraries to process images for various purposes. The script performs the following tasks:

1. Detects a blue rectangle in the image using color filtering techniques.
2. Identifies circles inside the detected blue rectangle and determines their colors.

## Features

- **Blue Rectangle Detection**: Identifies a blue rectangle using HSV color space filtering.
- **Circle and Color Detection**: Finds circles within the blue rectangle and detects their color.

## Installation

To set up and run this project, follow these steps:

### Prerequisites

- **Python 3.x**: Ensure you have Python 3 installed on your system. You can download it from the [official Python website](https://www.python.org/downloads/).

### Steps to Install

1. **Clone the Repository**:

    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2. **Create a virtual environment**:

    ```bash
    python3 -m venv venv
    ```

3. **Activate the virtual environment**:

    - On macOS and Linux:

      ```bash
      source venv/bin/activate
      ```

    - On Windows:

      ```bash
      .\venv\Scripts\activate
      ```

4. **Install the required dependencies**:

    Make sure you have a `requirements.txt` file in the project directory, then run:

    ```bash
    pip install -r requirements.txt
    ```



## Usage

After setting up the environment and installing the dependencies, you can run the demo script. Ensure you replace `'path/your/images'` in the script with the actual path to your image file.

```bash
python3 app.py


```bash
python3 webcam.py