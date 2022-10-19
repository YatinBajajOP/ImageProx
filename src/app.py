import streamlit as st
from functions import *

st.title("Welcome to ImageProx")
choices = ["Transformations", "Spatial Filtering", "Custom Editing", "App presets", "Other Functionalities", "Adding noise"]
choice = st.sidebar.selectbox("Choose the edits:", choices)


if choice == "Transformations":
    transformations = ["Log", "Power", "Cosine", "Fourier", "Haar"]
    transformation = st.sidebar.radio("Choose the transformations:", transformations, key="Transform")
    if transformation == "Log":
        st.header("Logarithmic Transformation")
        st.text('''
        Logarithmic transformation of an image is one of the gray level image 
transformations. 
Log transformation of an image means replacing all pixel values, present 
in the image, with its logarithmic values. 
Log transformation is used for image enhancement as it expands dark pixels
of the image as compared to higher pixel values.

The formula for applying log transformation in an image is,''')
        st.code('S = c * log (1 + r)')

        st.text('''
        where,
        R = input pixel value,
        C = scaling constant and
        S = output pixel value
        The value of 'c' is chosen such that we get the maximum output value 
        corresponding to the bit size used. So, the formula for calculating 'c' 
        is as follows:

        c = 255 / (log (1 + max_input_pixel_value))
                ''')
        st.write("Refer the code of log transformation below:")
        st.code('''
        import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read an image
image = cv2.imread('Sample .png')

# Apply log transformation method
c = 255 / np.log(1 + np.max(image))
log_image = c * (np.log(image + 1))

# Specify the data type so that
# float value will be converted to int
log_image = np.array(log_image, dtype = np.uint8)
        ''', language='python')
        img = st.file_uploader(label = "Upload the image") 
        # st.radio("Value of c", ) 
        path = storeImage(img, "log")
        c = -1
        if not st.checkbox("Default value of c (Scaling factor)"):
            c = st.slider("Value of c:", min_value=0, max_value=255, value=1)
        if img is not None:
            logTransform(path, c)

    elif transformation == "Power":
        st.header("Power Transformation")
        st.text('''Gamma correction is important for displaying images on a screen correctly, 
to prevent bleaching or darkening of images when viewed from different 
types of monitors with different display settings. This is done because 
our eyes pereive images in a gamma-shaped curve, whereas cameras capture
images in a linear fashion. Below is the Python code to apply gamma correction.

''')
        st.write("Power-law (gamma) transformations can be mathematically expressed as:")
        st.code("s = c * (r ** gamma)")
        st.write("Refer to the code below:")
        st.code('''import cv2
import numpy as np

# Open the image.
img = cv2.imread('sample.jpg')

# Trying 4 gamma values.
for gamma in [0.1, 0.5, 1.2, 2.2]:
	
# Apply gamma correction.
gamma_corrected = np.array(255*(img / 255) ** gamma, dtype = 'uint8')

# Save edited images.
cv2.imwrite('gamma_transformed'+str(gamma)+'.jpg', gamma_corrected''',language="python")
        img = st.file_uploader(label = "Upload the image") 
        # st.radio("Value of c", ) 
        path = storeImage(img, "power")
        c = -1
        if not st.checkbox("Default value of c (Scaling factor)"):
            c = st.slider("Value of c:", min_value=0, max_value=255, value=255)
        gamma = st.slider("Value of Gamma:", min_value=0.0, max_value=8.0, value=0.4)
        if img is not None:
            powerTransform(path, c, gamma)


elif choice == "Spatial Filtering":
    filters = ["Contrast", "Brightness", "Hue", "Smoothning - Mean", "Sharpening", "Median Filter"]
    filter = st.sidebar.radio("Spatial Filtering", filters, key="Filter")
    if filter == "Contrast":
        img = st.file_uploader(label = "Upload the image") 
        path = storeImage(img, "contrast")
        r1 = st.slider("Choose the value of r1 :", min_value=0, max_value=255, value=70)
        s1 = st.slider("Choose the value of s1 :", min_value=0, max_value=255, value=0)
        r2 = st.slider("Choose the value of r2 :", min_value=0, max_value=255, value=140)
        s2 = st.slider("Choose the value of s2 :", min_value=0, max_value=255, value=255)
        if img is not None:
            contrastStretching(path, r1, s1, r2, s2)
    


elif choice == "Custom Editing":
    edits = ["Crop", "Resize", "Rotate", "Shear"]
    edit = st.sidebar.radio("Custom Editing", edits, key="Edit")
    if edit == "Rotate":
        img = st.file_uploader(label = "Upload the image")
        path = storeImage(img, "rotate")
        degree = st.slider("Select the rotation angle", min_value=-180, max_value=180, value = 90)
        if img is not None:
            rotate(path, degree)
        
elif choice == "Other Functionalities":
    functionalities = ["PDF to Word", "Word to PDF", "JPEG to .txt", "PDF to .txt"]
    functionality = st.sidebar.radio("Functionalities", functionalities)

elif choice == "Adding noise":
    noises = ["Salt and pepper noise", "Squares and rectangles", "Gaussian", "Poisson", "Speckle"]
    noise = st.sidebar.radio("Noise addition", noises, key="Noise")
    if noise == "Salt and pepper noise":
        st.header("Salt and pepper noise")


        st.text(''' It is found only in grayscale images (black and white image). 
As the name suggests salt (white) in pepper (black)-white spots in
the dark regions or pepper (black) in salt (white)-black spots in
the white regions.
In other words, an image having salt-and-pepper noise will have a
few dark pixels in bright regions and a few bright pixels in dark 
regions. Salt-and-pepper noise is also called impulse noise. 
It can be caused by several reasons like dead pixels, 
analog-to-digital conversion error, bit transmission error, etc.''')

        st.write("Refer to the code below:")

        st.code('''import random
        import cv2

        def add_noise(img):

            # Getting the dimensions of the image
            row , col = img.shape
            
            # Randomly pick some pixels in the
            # image for coloring them white
            # Pick a random number between 300 and 10000
            number_of_pixels = random.randint(300, 10000)
            for i in range(number_of_pixels):
            
                # Pick a random y coordinate
                y_coord=random.randint(0, row - 1)
                
                # Pick a random x coordinate
                x_coord=random.randint(0, col - 1)
                
                # Color that pixel to white
                img[y_coord][x_coord] = 255
                
            # Randomly pick some pixels in
            # the image for coloring them black
            # Pick a random number between 300 and 10000
            number_of_pixels = random.randint(300 , 10000)
            for i in range(number_of_pixels):
            
                # Pick a random y coordinate
                y_coord=random.randint(0, row - 1)
                
                # Pick a random x coordinate
                x_coord=random.randint(0, col - 1)
                
                # Color that pixel to black
                img[y_coord][x_coord] = 0
                
            return img

        # salt-and-pepper noise can
        # be applied only to grayscale images
        # Reading the color image in grayscale image
        img = cv2.imread('lena.jpg',
                        cv2.IMREAD_GRAYSCALE)

        #Storing the image
        cv2.imwrite('salt-and-pepper-lena.jpg',
                    add_noise(img))
        ''',language="python")
        img = st.file_uploader(label = "Upload the image")
        path = storeImage(img, "saltnpepper")
        x, y = st.slider("Select the range of noise level",
                300, 10000, (500, 5000)
        )
        if img is not None:
            add_saltnpepper(path, x, y)

    elif noise == "Squares and rectangles":
        img = st.file_uploader(label = "Upload the image")
        path = storeImage(img, "boxes")
        x, y = st.slider("Select the range of sizes of boxes",
                0, 100, (20, 50)
        )
        num_boxes = st.slider("Select the number of boxes in the image",
                min_value=0, max_value=50, value = 10
        )
        if img is not None:
            add_boxes(path, num_boxes, x, y)


else:
    presets = ["Youtube Thumbnail", "Instagram post", "Instagram Story", "Passport size"]
    preset = st.sidebar.radio("App presets", presets, key="Preset")



    


