# Functions file
import cv2
import streamlit as st
from PIL import Image
import random
import numpy as np
from datetime import datetime, time



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

# Transformations
def logTransform(path, c):
    img = cv2.imread(path)
    if c == -1:
        c = 255 / np.log(1 + np.max(img))
    log_image = c * (np.log(img + 1))
    log_image = np.array(log_image, dtype = np.uint8)
    output_path = ".//static//output//log//" + path.split("//")[-1]
    cv2.imwrite(output_path, log_image)
    st.image(output_path,  caption="Logged Image")
    btn = st.download_button(
      label="Download image",
      data=output_path,
      file_name="log_" + path.split("//")[-1],
      mime="image/png")

def powerTransform(path, c, gamma):
    img = cv2.imread(path)
    if c == -1:
        c = 255 / np.log(1 + np.max(img))
    log_image = np.array(c*(img/c)**gamma,dtype='uint8')
    log_image = np.array(log_image, dtype = np.uint8)
    output_path = ".//static//output//power//" + path.split("//")[-1]
    cv2.imwrite(output_path, log_image)
    st.image(output_path,  caption="Powered Image")
    btn = st.download_button(
      label="Download image",
      data=output_path,
      file_name="log_" + path.split("//")[-1],
      mime="image/png")
    

# Translations
def rotate(path, degree):
    img = cv2.imread(path)
    M = cv2.getRotationMatrix2D((img.shape[1]/2, img.shape[0]/2), degree, 1) 
    rotated_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0])) 
    output_path = ".//static//output//rotate//" + path.split("//")[-1]
    cv2.imwrite(output_path, rotated_img)
    st.image(output_path,  caption="Rotated Image")
    btn = st.download_button(
      label="Download image",
      data=output_path,
      file_name="rotated_" + path.split("//")[-1],
      mime="image/png")


# Adding noise
def add_saltnpepper(path, x, y):
    img = cv2.imread(path)
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
    output_path = ".//static//output//saltnpepper//" + path.split("//")[-1]
    cv2.imwrite(output_path, img)
    st.image(output_path,  caption="Noisy Image")
    btn = st.download_button(
      label="Download image",
      data=output_path,
      file_name="saltnpepper_" + path.split("//")[-1],
      mime="image/png")

def add_boxes(path, num_boxes, size_x, size_y):
    img = cv2.imread(path)
    for i in range(np.random.randint(num_boxes)):
        box_x, box_y = np.random.randint(size_x,size_y, size=2)
        x, y = np.random.randint(0,max(img.shape[0], img.shape[1]) - max(box_x,box_y), size=2)
        img[x:x + box_x, y:y+box_y, :] = 255
    
    output_path = ".//static//output//boxes//" + path.split("//")[-1]
    cv2.imwrite(output_path, img)
    st.image(output_path,  caption="Boxy Image")
    btn = st.download_button(
      label="Download image",
      data=output_path,
      file_name="boxy_" + path.split("//")[-1],
      mime="image/png")

# Spatial Filtering
def pixelVal(pix, r1, s1, r2, s2):
    if (0 <= pix and pix <= r1):
        return (s1 / r1)*pix
    elif (r1 < pix and pix <= r2):
        return ((s2 - s1)/(r2 - r1)) * (pix - r1) + s1
    else:
        return ((255 - s2)/(255 - r2)) * (pix - r2) + s2

def contrastStretching(path, r1, s1, r2, s2):
    img = cv2.imread(path)
    pixelVal_vec = np.vectorize(pixelVal)
    contrast_stretched = pixelVal_vec(img, r1, s1, r2, s2)
    output_path = ".//static//output//contrast//" + path.split("//")[-1]
    cv2.imwrite(output_path, contrast_stretched)
    st.image(output_path,  caption="Contrasted Image")
    btn = st.download_button(
      label="Download image",
      data=output_path,
      file_name="contrast_" + path.split("//")[-1],
      mime="image/png")

def removeFileCheck(path, thresholdHours = 6):
    timeNow = datetime.now()
    timestr = datetime.fromtimestamp(os.path.getmtime(path)).strftime('%d,%m,%Y,%H,%M,%S')
    dateFile, monthFile, yearFile, hourFile, minutesFile, secondsFile = map(int, timestr.split(","))
    if (datetime.now().year) > yearFile or datetime.now().month > monthFile or datetime.now().day > dateFile:
        return True
    elif datetime.now().hour - hourFile > thresholdHours:
        return True
    return False

# Other functionalities
def ImageToText(path):
    import pytesseract   
    


