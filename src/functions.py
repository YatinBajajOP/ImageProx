# Functions file
import cv2
import streamlit as st
from PIL import Image, ImageFilter
import random
import numpy as np
from datetime import datetime
import os
import skimage
from scipy.fftpack import idct, dct
from skimage.io import imread


# Data management
def storeImage(image_file, operation):
    if image_file is not None:
        file_details = {"FileName":image_file.name,"FileType":image_file.type}
        # print(file_details)
        path = ".//static//input//" + operation + "//" + image_file.name
        with open(path,"wb") as f: 
            f.write(image_file.getbuffer())         
        # st.success("Saved File")
        return path

def saveOutput(img, label, path, caption, file_prefix):
    output_path = ".//static//output//" + label + "//" + path.split("//")[-1]
    cv2.imwrite(output_path, img)
    st.image(output_path,  caption=caption)
    # st.write(output_path)
    with open(output_path, "rb") as file:
        btn = st.download_button(
            label="Download image",
            data=file,
            file_name=file_prefix + path.split("//")[-1],
            mime="image/jpg")


# Transformations
def logTransform(path, c):
    img = cv2.imread(path, flags=cv2.IMREAD_COLOR)
    if c == -1:
        c = 255 / np.log(1 + np.max(img))
    log_image = c * (np.log(img + 1))
    log_image = np.array(log_image, dtype = np.uint8)
    saveOutput(log_image, "log", path, "Logged Image", "log_")

def powerTransform(path, c, gamma):
    img = cv2.imread(path, flags=cv2.IMREAD_COLOR)
    if c == -1:
        c = 255 / np.log(1 + np.max(img))
    power_image = np.array(c*(img/c)**gamma,dtype='uint8')
    power_image = np.array(power_image, dtype = np.uint8)
    saveOutput(power_image, "power", path, "Powered Image", "power_")

# For cosine transformation
# implement 2D DCT
def dct2(a):
    return dct(dct(a.T, norm='ortho').T, norm='ortho')

# implement 2D IDCT
def idct2(a):
    return idct(idct(a.T, norm='ortho').T, norm='ortho')    

def cosineTransform(path):
    im = imread(path) 
    imF = dct2(im)
    im1 = idct2(imF)
    im1 = im1
    saveOutput(im1, "cosine", path, "Cosine Transformed Image", "cosine_")


# Translations
def rotate(path, degree):
    img = cv2.imread(path, flags=cv2.IMREAD_COLOR)
    M = cv2.getRotationMatrix2D((img.shape[1]/2, img.shape[0]/2), degree, 1) 
    rotated_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0])) 
    saveOutput(rotated_img, "rotate", path, "Rotated Image", "rotated_")

def resize(path, x, y):
    img = Image.open(path)
    img = img.resize((x, y), Image.ANTIALIAS)
    img = np.asarray(img)
    saveOutput(img, "resize", path, "Resized image", "resized_")

def crop(path, x1, y1, x2, y2):
    img = Image.open(path)
    area = (x1, y1, x2, y2)
    img = img.crop(area)
    img = np.asarray(img)
    saveOutput(img, "crop", path, "Cropped Image", "cropped_")

# Adding noise
def add_saltnpepper(path, x, y):
    img = cv2.imread(path, flags=cv2.IMREAD_COLOR)
    row , col, levels = img.shape
    number_of_pixels = random.randint(x, y)
    for i in range(number_of_pixels):
        y_coord=random.randint(0, row - 1)
        x_coord=random.randint(0, col - 1)
        img[y_coord][x_coord] = 255
    number_of_pixels = random.randint(x , y)
    for i in range(number_of_pixels):
        y_coord=random.randint(0, row - 1)
        x_coord=random.randint(0, col - 1)
        img[y_coord][x_coord] = 0
    saveOutput(img, "saltnpepper", path, "Image with salt and pepper", "saltnpepper_")

def add_gaussian(path):
    img = skimage.io.imread(path)/255.0
    gimg = skimage.util.random_noise(img, mode="gaussian")
    gimg = gimg * 255.0
    saveOutput(gimg, "gaussian", path, "Image with gaussian noise", "gaussian_")

def add_localvar(path):
    img = skimage.io.imread(path)/255.0
    gimg = skimage.util.random_noise(img, mode="localvar")
    gimg = gimg * 255.0
    saveOutput(gimg, "localvar", path, "Image with localvar noise", "localvar_")

def add_speckle(path):
    img = skimage.io.imread(path)/255.0
    gimg = skimage.util.random_noise(img, mode="speckle")
    gimg = gimg * 255.0
    saveOutput(gimg, "speckle", path, "Image with speckle noise", "speckle_")

def add_pepper(path):
    img = skimage.io.imread(path)/255.0
    gimg = skimage.util.random_noise(img, mode="pepper")
    gimg = gimg * 255.0
    saveOutput(gimg, "pepper", path, "Image with pepper noise", "pepper_")

def add_salt(path):
    img = skimage.io.imread(path)/255.0
    gimg = skimage.util.random_noise(img, mode="salt")
    gimg = gimg * 255.0
    saveOutput(gimg, "salt", path, "Image with salt noise", "salt_")

def add_poisson(path):
    img = skimage.io.imread(path)/255.0
    gimg = skimage.util.random_noise(img, mode="poisson")
    gimg = gimg * 255.0
    saveOutput(gimg, "poisson", path, "Image with poisson noise", "poisson_")

def add_sandp(path):
    img = skimage.io.imread(path)/255.0
    gimg = skimage.util.random_noise(img, mode="s&p")
    gimg = gimg * 255.0
    saveOutput(gimg, "sandp", path, "Image with s&p noise", "sandp_")

def add_boxes(path, num_boxes, size_x, size_y):
    img = cv2.imread(path, flags=cv2.IMREAD_COLOR)
    for i in range(np.random.randint(num_boxes)):
        box_x, box_y = np.random.randint(size_x,size_y, size=2)
        x, y = np.random.randint(0,max(img.shape[0], img.shape[1]) - max(box_x,box_y), size=2)
        img[x:x + box_x, y:y+box_y, :] = 255
    saveOutput(img, "boxes", path, "Boxy Image", "boxy_")


# Spatial Filtering
def pixelVal(pix, r1, s1, r2, s2):
    if (0 <= pix and pix <= r1):
        return (s1 / r1)*pix
    elif (r1 < pix and pix <= r2):
        return ((s2 - s1)/(r2 - r1)) * (pix - r1) + s1
    else:
        return ((255 - s2)/(255 - r2)) * (pix - r2) + s2

def contrastStretching(path, r1, s1, r2, s2):
    img = cv2.imread(path, flags=cv2.IMREAD_COLOR)
    pixelVal_vec = np.vectorize(pixelVal)
    contrast_stretched = pixelVal_vec(img, r1, s1, r2, s2)
    saveOutput(contrast_stretched, "contrast", path, "Contrast stretched image", "contrast_")

def removeFileCheck(path, thresholdHours = 6):
    timestr = datetime.fromtimestamp(os.path.getmtime(path)).strftime('%d,%m,%Y,%H,%M,%S')
    dateFile, monthFile, yearFile, hourFile, minutesFile, secondsFile = map(int, timestr.split(","))
    if (datetime.now().year) > yearFile or datetime.now().month > monthFile or datetime.now().day > dateFile:
        return True
    elif datetime.now().hour - hourFile > thresholdHours:
        return True
    return False

def sharpeningImage(path, level=5):
    img = cv2.imread(path, flags=cv2.IMREAD_COLOR)
    kernel = np.array([[0, -1, 0],
                   [-1, level,-1],
                   [0, -1, 0]])
    image_sharp = cv2.filter2D(src=img, ddepth=-1, kernel=kernel)
    saveOutput(image_sharp, "sharpen", path, "Sharpened Image", "sharpened_")

def blurringImage(path):
    img = cv2.imread(path, flags=cv2.IMREAD_COLOR)
    kernel = (1/16.0) * np.array([[1., 2., 1.],
                                  [2., 4., 2.],
                                  [1., 2., 1.]])
    image_sharp = cv2.filter2D(src=img, ddepth=-1, kernel=kernel)
    saveOutput(image_sharp, "blur", path, "Blurred Image", "blurred_")

def medianFilter(path, size):
    img = Image.open(path)
    im2 = np.asarray(img.filter(ImageFilter.MedianFilter(size = size)))
    saveOutput(im2, "median", path, "Median filtered image", "median_") 

def modeFilter(path, size):
    img = Image.open(path)
    im2 = np.asarray(img.filter(ImageFilter.ModeFilter(size = size)))
    saveOutput(im2, "mode", path, "Mode filtered image", "mode_") 


