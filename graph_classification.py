import os
import time
import numpy as np

from PIL import Image
from keras.preprocessing import image
from tensorflow.keras.models import load_model

# import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")

# load inception resnet50 model
print("[INFO] Loading classfication model...")
vertical_bar_model = load_model("cls_model/vertical_bar_classification_model.hdf5")
# horizontal_bar_model = load_model("cls_model/horizontal_bar_classification_model.hdf5")
# pie_chart_model = load_model("cls_model/")

def load_image_from_path(img_path_, show=False):
    img = image.load_img(img_path_, target_size=(224, 224))

    img_tensor = image.img_to_array(img)  # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor,
                                axis=0)  # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.  # imshow expects values in the range [0, 1]

    return img_tensor


def load_image_from_array(crop_img):
    newsize = (224, 224)
    img = Image.fromarray(crop_img)
    img = img.resize(newsize)

    img_tensor = image.img_to_array(img)  # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor,
                                axis=0)  # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.  # imshow expects values in the range [0, 1]

    return img_tensor


# Only one class will be returned.
def graph_classifation_pipeline(crop_img, graph_type):
    graph_details = dict()

    if graph_type == "Vertical Bar Graph":
        CATEGORIES_ = ["grouped_bar", "simple_bar", "stacked_bar"]

        # Loading image
        new_image_ = load_image_from_array(crop_img)

        # check prediction
        pred_ = vertical_bar_model.predict(new_image_)

        # obtain categories based on values
        pred_ = CATEGORIES_[np.argmax(pred_)]

    if graph_type == "Horizontal Bar Graph":
        CATEGORIES_ = ["grouped_bar", "simple_bar", "stacked_bar"]

        # Loading image
        new_image_ = load_image_from_array(crop_img)

        # check prediction
        pred_ = horizontal_bar_model.predict(new_image_)

        # obtain categories based on values
        pred_ = CATEGORIES_[np.argmax(pred_)]
        
    if graph_type == "Pie chart":
        pass

    if graph_type == "Line chart":
        pass

    graph_details[pred_] = crop_img[0]

    return graph_details


if __name__ == "__main__":
    # image path
    img_path = 'test/grpd.png'
    CATEGORIES = ["grouped_bar", "simple_bar", "stacked_bar"]

    # load a single image for testing
    new_image = load_image_from_path(img_path)
    pred = model.predict(new_image)
    pred = CATEGORIES[np.argmax(pred)]
    print("[INFO] Predicted graph:", pred)

    # s_time = time.time()
    # Prediction
    # graph_type = graph_classifation_pipeline(new_image)
    # print("[INFO] Total time taken to classify the image", time.time() - s_time)
    # print(graph_type)
