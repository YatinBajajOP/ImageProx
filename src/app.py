import streamlit as st
from functions import *
import os
from datetime import datetime, time

# Streamlit
st.set_page_config(page_icon="📸", page_title="ImageProx", layout="wide")
st.sidebar.title("ImageProx 📸")
st.markdown('<center><h1 style="color:red;">Welcome to ImageProx 📸</h1></center>', unsafe_allow_html=True)

choices = ["Transformations", "Spatial Filtering", "Custom Editing", "Adding noise"]
choice = st.sidebar.selectbox("Choose the edits:", choices)

if choice == "Transformations":
    transformations = ["Log", "Power", "Cosine"]
    transformation = st.sidebar.radio("Choose the transformations:", transformations, key="Transform")
    if transformation == "Log":
        st.header("Logarithmic Transformation")
        st.text('''
        Logarithmic transformation of an image is one of the gray level image transformations. 
Log transformation of an image means replacing all pixel values, present in the image, with its logarithmic values. 
Log transformation is used for image enhancement as it expands dark pixelsof the image as compared to higher pixel values.

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
        c = -1
        if not st.checkbox("Default value of c (Scaling factor)"):
            c = st.slider("Value of c:", min_value=0, max_value=255, value=1)
        if img is not None:
            path = storeImage(img, "log")
            logTransform(path, c)

    elif transformation == "Power":
        st.header("Power Transformation")
        st.text('''Gamma correction is important for displaying images on a screen correctly, to prevent bleaching or darkening of images when viewed from
different types of monitors with different display settings. This is done because our eyes pereive images in a gamma-shaped curve,
whereas cameras capture images in a linear fashion. Below is the Python code to apply gamma correction.

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
cv2.imwrite('gamma_transformed'+str(gamma)+'.jpg', gamma_corrected)''',language="python")
        img = st.file_uploader(label = "Upload the image") 
        # st.radio("Value of c", ) 
        c = -1
        if not st.checkbox("Default value of c (Scaling factor)"):
            c = st.slider("Value of c:", min_value=0, max_value=255, value=255)
        gamma = st.slider("Value of Gamma:", min_value=0.0, max_value=8.0, value=0.4)
        if img is not None:
            path = storeImage(img, "power")
            powerTransform(path, c, gamma)

    elif transformation == "Cosine":
        st.header("Cosine Transformation")
        st.text(''' Contrast stretching (often called normalization) is a simple image enhancement technique that attempts to improve the contrast in an image 
by 'stretching' the range of intensity values it contains to span a desired range of values, the full range of pixel values that the image
type concerned allows.It is a technique that tries to improve the image.
It improves the contrast in an image by stretching the intensity value range. It is the difference between the maximum and minimum intensity 
value in an image.''')

        st.write("Refer to the code below:")
        st.code('''from scipy.fftpack import dct, idct

# implement 2D DCT
def dct2(a):
    return dct(dct(a.T, norm='ortho').T, norm='ortho')

# implement 2D IDCT
def idct2(a):
    return idct(idct(a.T, norm='ortho').T, norm='ortho')    

from skimage.io import imread
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pylab as plt

# read lena RGB image and convert to grayscale
im = rgb2gray(imread('images/lena.jpg')) 
imF = dct2(im)
im1 = idct2(imF)

# check if the reconstructed image is nearly equal to the original image
np.allclose(im, im1)
# True

# plot original and reconstructed images with matplotlib.pylab
plt.gray()
plt.subplot(121), plt.imshow(im), plt.axis('off'), plt.title('original image', size=20)
plt.subplot(122), plt.imshow(im1), plt.axis('off'), plt.title('reconstructed image (DCT IDCT)', size=20)
plt.show() ''',
language="python")
        img = st.file_uploader(label = "Upload the image")
        if img is not None:
            path = storeImage(img, "cosine")
            cosineTransform(path) 

elif choice == "Spatial Filtering":
    filters = ["Contrast", "Smoothning - Mean", "Sharpening", "Median Filter", "Mode Filter"]
    filter = st.sidebar.radio("Spatial Filtering", filters, key="Filter")
    if filter == "Contrast":
        st.header("Contrast Stretching")
        st.text(''' Contrast stretching (often called normalization) is a simple image enhancement technique that attempts to improve the contrast in an image by 'stretching' the range of intensity values it contains to span a desired range of values, the full range of pixel values that the image type concerned allows.It is a technique that tries to improve the image.
It improves the contrast in an image by stretching the intensity value range.It is the difference between the maximum and minimum intensity value in an image.''')

        st.write("Refer to the code below:")
        st.code('''import cv2
importnumpy as np
fromIPython.display import Image
frommatplotlib import pyplot as plt
frommatplotlib import pyplot as plt1

frommatplotlib import pyplot as plt2

r1=80
r2=200
s1=30
s2=210
img = cv2.imread("buoy.jpg",0)
al=s1/r1
bt=(s2-s1)/(r2-r1)
gm=(255-s2)/(255-r2)
c1=s1-bt*r1
c2=s2-gm*r2
d=img
fori in range(len(img)):
for j in range(len(img[0])):
if(img[i][j] < r1):
d[i][j] = al*img[i][j]
elif( r1 <= img[i][j] < r2 ):
d[i][j] = bt*img[i][j]   c1
else:
d[i][j] = gm*img[i][j]   c2
cv2.imwrite('buoy_contrast.jpg',d)
Image('buoy.jpg')
Image('buoy_contrast.jpg') ''',
language="python")
        img = st.file_uploader(label = "Upload the image") 
        r1 = st.slider("Choose the value of r1 :", min_value=0, max_value=255, value=70)
        s1 = st.slider("Choose the value of s1 :", min_value=0, max_value=255, value=0)
        r2 = st.slider("Choose the value of r2 :", min_value=0, max_value=255, value=140)
        s2 = st.slider("Choose the value of s2 :", min_value=0, max_value=255, value=255)
        if img is not None:
            path = storeImage(img, "contrast")
            contrastStretching(path, r1, s1, r2, s2)

    elif filter == "Sharpening":
        img = st.file_uploader(label = "Upload the image") 
        if img is not None:
            path = storeImage(img, "sharpen")
            sharpeningImage(path)

    elif filter == "Smoothning - Mean":
        img = st.file_uploader(label = "Upload the image") 
        if img is not None:
            path = storeImage(img, "blur")
            blurringImage(path)

    elif filter == "Median Filter":
        img = st.file_uploader(label = "Upload the image") 
        size = st.slider(label="Select the size of the filter", min_value=1, max_value=15, value=3, step=2)
        if img is not None:
            path = storeImage(img, "median")
            medianFilter(path, size)

    elif filter == "Mode Filter":
        img = st.file_uploader(label = "Upload the image") 
        size = st.slider(label="Select the size of the filter", min_value=1, max_value=15, value=3, step=2)
        if img is not None:
            path = storeImage(img, "mode")
            modeFilter(path, size)
    
elif choice == "Custom Editing":
    edits = ["Crop", "Resize", "Rotate"]
    edit = st.sidebar.radio("Custom Editing", edits, key="Edit")
    img = st.file_uploader(label = "Upload the image")

    if edit == "Rotate":
        degree = st.slider("Select the rotation angle", min_value=-180, max_value=180, value = 90)
        if img is not None:
            path = storeImage(img, "rotate")
            rotate(path, degree)

    elif edit == "Resize":
        width = st.number_input("Enter the width : ", min_value=0, max_value=1000, value=50, step=1)
        height = st.number_input("Enter the height : ", min_value=0, max_value=1000, value=50, step=1)
        if img is not None:
            path = storeImage(img, "resize")
            resize(path, width, height)
    
    elif edit == "Crop":
        x1 = st.number_input("Enter the value of x1", min_value=0, max_value=1000, value=0, step=1)
        x2 = st.number_input("Enter the value of x2", min_value=0, max_value=1000, value=0, step=1)
        y1 = st.number_input("Enter the value of y1", min_value=0, max_value=1000, value=50, step=1)
        y2 = st.number_input("Enter the value of y2", min_value=0, max_value=1000, value=50, step=1)
        if img is not None:
            path = storeImage(img, "resize")
            crop(path, x1, x2, y1, y2)



elif choice == "Adding noise":
    noises = ["Salt and pepper noise", "Squares and rectangles", "Gaussian", "Poisson", "Speckle", "Localvar", "Pepper", "Salt", "Salt&Pepper"]
    noise = st.sidebar.radio("Noise addition", noises, key="Noise")
    if noise == "Salt and pepper noise":
        st.header("Salt and pepper noise")
        st.text(''' It is found only in grayscale images (black and white image). As the name suggests salt (white) in pepper (black)-white spots in
the dark regions or pepper (black) in salt (white)-black spots in the white regions.
In other words, an image having salt-and-pepper noise will have a few dark pixels in bright regions and a few bright pixels in dark 
regions. Salt-and-pepper noise is also called impulse noise. It can be caused by several reasons like dead pixels, 
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
        x, y = st.slider("Select the range of noise level",
                300, 10000, (500, 5000)
        )
        if img is not None:
            path = storeImage(img, "saltnpepper")
            add_saltnpepper(path, x, y)

    elif noise == "Squares and rectangles":
        img = st.file_uploader(label = "Upload the image")
        x, y = st.slider("Select the range of sizes of boxes",
                0, 100, (20, 50)
        )
        num_boxes = st.slider("Select the number of boxes in the image",
                min_value=0, max_value=50, value = 10
        )
        if img is not None:
            path = storeImage(img, "boxes")
            add_boxes(path, num_boxes, x, y)

    elif noise == "Gaussian":
        img = st.file_uploader(label = "Upload the image")
        if img is not None:
            path = storeImage(img, "gaussian")
            add_gaussian(path)
    
    elif noise == "Poisson":
        img = st.file_uploader(label = "Upload the image")
        if img is not None:
            path = storeImage(img, "poisson")
            add_poisson(path)

    elif noise == "Speckle":
        img = st.file_uploader(label = "Upload the image")
        if img is not None:
            path = storeImage(img, "speckle")
            add_speckle(path)

    elif noise == "Pepper":
        img = st.file_uploader(label = "Upload the image")
        if img is not None:
            path = storeImage(img, "pepper")
            add_pepper(path)
    
    elif noise == "Salt":
        img = st.file_uploader(label = "Upload the image")
        if img is not None:
            path = storeImage(img, "salt")
            add_salt(path)

    elif noise == "Salt&Pepper":
        img = st.file_uploader(label = "Upload the image")
        if img is not None:
            path = storeImage(img, "sandp")
            add_sandp(path)

    elif noise == "Localvar":
        img = st.file_uploader(label = "Upload the image")
        if img is not None:
            path = storeImage(img, "localvar")
            add_localvar(path)


if st.sidebar.button("Clear cache"):
    root = "./static/"
    for path, subdirs, files in os.walk(root):
        for name in files:  
            newPath = path + "/" + name
            newPath = newPath.replace("\\","/")
            if removeFileCheck(newPath):
                os.remove(newPath)