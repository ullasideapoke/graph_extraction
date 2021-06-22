'''
pip install deskew
pip install opencv-python
apt install tesseract-ocr
apt install libtesseract-dev
pip install Pillow
pip install pytesseract
'''

import deskew
from deskew import determine_skew
from skimage.transform import rotate
import numpy as np
from skimage import io
from skimage.color import rgb2gray
from skimage.transform import rotate
from deskew import determine_skew
import math
import cv2
import numpy as np
import pytesseract
from PIL import ImageEnhance, ImageFilter, Image
from PIL import Image
import pytesseract

def angle_prediction(image):
    grayscale = rgb2gray(image)
    angle = determine_skew(grayscale)
    
    return angle

def rotate_image(image, angle):
    image_size = (image.shape[1], image.shape[0])
    image_center = tuple(np.array(image_size) / 2)

    rot_mat = np.vstack(
        [cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]]
    )

    rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

    # Shorthand for below calcs
    image_w2 = image_size[0] * 0.5
    image_h2 = image_size[1] * 0.5

    # Obtain the rotated coordinates of the image corners
    rotated_coords = [
        (np.array([-image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([ image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],
        (np.array([ image_w2, -image_h2]) * rot_mat_notranslate).A[0]
    ]

    # Find the size of the new image
    x_coords = [pt[0] for pt in rotated_coords]
    x_pos = [x for x in x_coords if x > 0]
    x_neg = [x for x in x_coords if x < 0]

    y_coords = [pt[1] for pt in rotated_coords]
    y_pos = [y for y in y_coords if y > 0]
    y_neg = [y for y in y_coords if y < 0]

    right_bound = max(x_pos)
    left_bound = min(x_neg)
    top_bound = max(y_pos)
    bot_bound = min(y_neg)

    new_w = int(abs(right_bound - left_bound))
    new_h = int(abs(top_bound - bot_bound))

    # We require a translation matrix to keep the image centred
    trans_mat = np.matrix([
        [1, 0, int(new_w * 0.5 - image_w2)],
        [0, 1, int(new_h * 0.5 - image_h2)],
        [0, 0, 1]
    ])

    # Compute the tranform for the combined rotation and translation
    affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]

    # Apply the transform
    result = cv2.warpAffine(
        image,
        affine_mat,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR
    )

    return result

def ocr_data_extracted(img__):
#    This function will handle the core OCR processing of images.
    text = pytesseract.image_to_string(img__)
    text = " ".join([i for i in text.split("\n") if len(i) > 0])
    
    return text


def y_axis_title_extraction_pipeline(img_):
    ang = angle_prediction(img_)
    rotated_image = rotate_image(img_, ang) * 255
    rotated_image = rotated_image.astype(np.uint8)
    text = ocr_data_extracted(rotated_image)
    
    return text


def x_axis_title_extraction_pipeline(img_):
    text = ocr_data_extracted(img_)

    return text

def graph_title_extraction_pipeline(img_):
    text = ocr_data_extracted(img_)

    return text

def image_rotation(image):
    ang = angle_prediction(image)
    rotated_image = rotate_image(image, ang) * 255
    rotated_image = rotated_image.astype(np.uint8)

    return rotated_image  

if __name__=='__main__':
    img = io.imread('/home/pushpendu/Documents/All_task/market_related_information/image_data_extraction/swasthik_ajith_res/Reg _ Vertical bar graph/Y_axis_title_Extract/y_title.jpg')
    text = y_axis_data_extraction_pipeline(img)
    print(text)