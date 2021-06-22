# import the necessary packages
import os
import cv2
import time
import numpy as np

import warnings
warnings.filterwarnings("ignore")

# load the class labels our YOLO model was trained on custom dataset
vertical_simple_labelsPath = "cfg/graph_feature/vertical/simple_bar_graph_feature_detection.names"
vertical_grouped_labelsPath = "cfg/graph_feature/vertical/verti_group_bar_graph_feature_detection.names"
vertical_stacked_labelsPath = "cfg/graph_feature/vertical/verti_stacked_bar_graph_feature_detection.names"

horizontal_simple_labelsPath = "cfg/graph_feature/horizontal/simple_bar_graph_feature_detection.names"
horizontal_grouped_labelsPath = "cfg/graph_feature/horizontal/grouped_bar_graph_feature_detection.names"
horizontal_stacked_labelsPath = "cfg/graph_feature/horizontal/stacked_bar_graph_feature_detection.names"


vertical_simple_LABELS = open(vertical_simple_labelsPath).read().strip().split("\n")
vertical_stacked_LABELS = open(vertical_stacked_labelsPath).read().strip().split("\n")
vertical_grouped_LABELS = open(vertical_grouped_labelsPath).read().strip().split("\n")
horizontal_simple_LABELS = open(horizontal_simple_labelsPath).read().strip().split("\n")
horizontal_stacked_LABELS = open(horizontal_stacked_labelsPath).read().strip().split("\n")
horizontal_grouped_LABELS = open(horizontal_grouped_labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
vertical_simple_COLORS = np.random.randint(0, 255, size=(len(vertical_simple_LABELS), 3), dtype="uint8")
print("[INFO] Total number of classes for vertical simple chart", len(vertical_simple_COLORS))

vertical_stacked_COLORS = np.random.randint(0, 255, size=(len(vertical_stacked_LABELS), 3), dtype="uint8")
print("[INFO] Total number of classes for vertical stacked chart", len(vertical_stacked_COLORS))

vertical_grouped_COLORS = np.random.randint(0, 255, size=(len(vertical_grouped_LABELS), 3), dtype="uint8")
print("[INFO] Total number of classes for vertical grouped chart", len(vertical_grouped_COLORS))

horizontal_simple_COLORS = np.random.randint(0, 255, size=(len(horizontal_simple_LABELS), 3), dtype="uint8")
print("[INFO] Total number of classes for horizontal simple chart", len(horizontal_simple_COLORS))

horizontal_stacked_COLORS = np.random.randint(0, 255, size=(len(horizontal_stacked_LABELS), 3), dtype="uint8")
print("[INFO] Total number of classes for horizontal stacked chart", len(horizontal_stacked_COLORS))

horizontal_grouped_COLORS = np.random.randint(0, 255, size=(len(horizontal_grouped_LABELS), 3), dtype="uint8")
print("[INFO] Total number of classes for horizontal grouped chart", len(horizontal_grouped_COLORS))


# derive the paths to the YOLO weights and model configuration
# vertical_simple_weightsPath = "weights/graph_feature/vertical/simple_bar_graph_feature_detection_6000.weights"
vertical_grouped_weightsPath = "weights/graph_feature/vertical/verti_group_bar_graph_feature_detection_8000.weights"
# vertical_stacked_weightsPath = "weights/graph_feature/vertical/verti_stacked_bar_graph_feature_detection_14000.weights"


# horizontal_simple_weightsPath = "weights/graph_feature/horizontal/simple_bar_graph_feature_detection_6000.weights"
# horizontal_grouped_weightsPath = "weights/graph_feature/horizontal/grouped_bar_graph_feature_detection_10000.weights"
# horizontal_stacked_weightsPath = "weights/graph_feature/horizontal/stacked_bar_graph_feature_detection_8000.weights"


# configPath
# vertical_simple_configPath = "cfg/graph_feature/vertical/simple_bar_graph_feature_detection.cfg"
vertical_grouped_configPath = "cfg/graph_feature/vertical/verti_group_bar_graph_feature_detection.cfg"
# vertical_stacked_configPath = "cfg/graph_feature/vertical/verti_stacked_bar_graph_feature_detection.cfg"


# horizontal_simple_configPath = "cfg/graph_feature/horizontal/simple_bar_graph_feature_detection.cfg"
# horizontal_grouped_configPath = "cfg/graph_feature/horizontal/grouped_bar_graph_feature_detection.cfg"
# horizontal_stacked_configPath = "cfg/graph_feature/horizontal/stacked_bar_graph_feature_detection.cfg"


# load our YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO weight from disk...")
# vertical_simple_net = cv2.dnn.readNetFromDarknet(vertical_simple_configPath, vertical_simple_weightsPath)
vertical_grouped_net = cv2.dnn.readNetFromDarknet(vertical_grouped_configPath, vertical_grouped_weightsPath)
# vertical_stacked_net = cv2.dnn.readNetFromDarknet(vertical_stacked_configPath, vertical_stacked_weightsPath)
# horizontal_simple_net = cv2.dnn.readNetFromDarknet(horizontal_simple_configPath, horizontal_simple_weightsPath)
# horizontal_grouped_net = cv2.dnn.readNetFromDarknet(horizontal_grouped_configPath, horizontal_grouped_weightsPath)
# horizontal_stacked_net = cv2.dnn.readNetFromDarknet(horizontal_stacked_configPath, horizontal_stacked_weightsPath)

CONFIDENCE_THRESHOLD = 0.3

def model_prediction(img, net):
    # load our input image and grab its spatial dimensions
    (H, W) = img.shape[:2]

    # determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    print("[INFO] YOLO took {:.6f} seconds".format(end - start))

    return layerOutputs, W, H, img


def bounding_box_extraction(layerOutputs, W, H):
    boxes = []
    confidences = []
    classIDs = []

    # loop over each of the layer outputs
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > CONFIDENCE_THRESHOLD:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    return boxes, confidences, classIDs


def image_croping(boxes, confidences, classIDs, image_, LABELS_):
    legend_color_count = 0
    legend_text_count = 0
    stack_count = 0
    bar_count = 0
    graph_feature_image = dict()
    
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, CONFIDENCE_THRESHOLD)

    if len(idxs) > 0:
        for i in idxs.flatten():

            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            text = "{}: {:.4f}".format(LABELS_[classIDs[i]], confidences[i])
            
            im_rgb = cv2.cvtColor(image_, cv2.COLOR_BGR2RGB)
            crop_img = im_rgb[y:y + h, x:x + w]
            
            key_text = text.split(":")[0].strip()
            print("[INFO] key_text:", key_text)
            
            if key_text == 'bar':
                bar_count = bar_count + 1
                key_text = key_text + str(bar_count)
            if key_text == 'stack':
                stack_count = stack_count + 1
                key_text = key_text + str(stack_count)
            if key_text == 'legend_text':
                legend_text_count = legend_text_count + 1
                key_text = key_text + str(legend_text_count)
            if key_text == 'legend_color':
                legend_color_count = legend_color_count + 1
                key_text = key_text + str(legend_color_count)
            
            graph_feature_image[key_text] = [crop_img, boxes[i]]

    return graph_feature_image


# Feature detection method
def graph_feature_detection_pipeline(image_, graph_image, graph_sub_type):

    if graph_image == "Vertical Bar Graph":
        
        if graph_sub_type == "simple_bar":
            layerOutputs = model_prediction(image_, vertical_simple_net)
            boxes, confidences, classIDs = bounding_box_extraction(layerOutputs[0], layerOutputs[1], layerOutputs[2])
            image_ = layerOutputs[3]
            image_croping_res_ = image_croping(boxes, confidences, classIDs, image_, vertical_simple_LABELS)

        if graph_sub_type == "stacked_bar":
            layerOutputs = model_prediction(image_, vertical_stacked_net)
            boxes, confidences, classIDs = bounding_box_extraction(layerOutputs[0], layerOutputs[1], layerOutputs[2])
            image_ = layerOutputs[3]
            image_croping_res_ = image_croping(boxes, confidences, classIDs, image_, vertical_stacked_LABELS)

        if graph_sub_type == "grouped_bar":
            layerOutputs = model_prediction(image_, vertical_grouped_net)
            boxes, confidences, classIDs = bounding_box_extraction(layerOutputs[0], layerOutputs[1], layerOutputs[2])
            image_ = layerOutputs[3]
            image_croping_res_ = image_croping(boxes, confidences, classIDs, image_, vertical_grouped_LABELS)

    if graph_image == "Horizontal Bar Graph":
        
        if graph_sub_type == "simple_bar":
            layerOutputs = model_prediction(image_, horizontal_simple_net)
            boxes, confidences, classIDs = bounding_box_extraction(layerOutputs[0], layerOutputs[1], layerOutputs[2])
            image_ = layerOutputs[3]
            image_croping_res_ = image_croping(boxes, confidences, classIDs, image_, horizontal_simple_LABELS)

        if graph_sub_type == "stacked_bar":
            layerOutputs = model_prediction(image_, horizontal_stacked_net)
            boxes, confidences, classIDs = bounding_box_extraction(layerOutputs[0], layerOutputs[1], layerOutputs[2])
            image_ = layerOutputs[3]
            image_croping_res_ = image_croping(boxes, confidences, classIDs, image_, horizontal_stacked_LABELS)

        if graph_sub_type == "grouped_bar":
            layerOutputs = model_prediction(image_, horizontal_grouped_net)
            boxes, confidences, classIDs = bounding_box_extraction(layerOutputs[0], layerOutputs[1], layerOutputs[2])
            image_ = layerOutputs[3]
            image_croping_res_ = image_croping(boxes, confidences, classIDs, image_, horizontal_grouped_LABELS)

    if graph_image == "Pie chart":
        pass

    if graph_image == "Line chart":
        pass

    return image_croping_res_


if __name__ == "__main__":
    path = "/home/pushpendu/Documents/All_task/market_related_information/image_data_extraction/horizontal_bar_chart/training_data/simple/cropped_pdf_images25_26_1.jpg"
    graph_image_ = "Horizontal Bar Graph"
    graph_sub_type_ = "simple_bar"
    
    image_ = cv2.imread(path)
    s_time = time.time()

    if graph_image_ == "Vertical Bar Graph":
        
        if graph_sub_type_ == "simple_bar":
            image_croping_res = graph_feature_detection_pipeline(image_, graph_image_, graph_sub_type_)

        if graph_sub_type_ == "stacked_bar":
            image_croping_res = graph_feature_detection_pipeline(image_, graph_image_, graph_sub_type_)

        if graph_sub_type_ == "grouped_bar":
            image_croping_res = graph_feature_detection_pipeline(image_, graph_image_, graph_sub_type_)

    if graph_image_ == "Horizontal Bar Graph":

        if graph_sub_type_ == "simple_bar":
            image_croping_res = graph_feature_detection_pipeline(image_, graph_image_, graph_sub_type_)

        if graph_sub_type_ == "stacked_bar":
            image_croping_res = graph_feature_detection_pipeline(image_, graph_image_, graph_sub_type_)

        if graph_sub_type_ == "grouped_bar":
            image_croping_res = graph_feature_detection_pipeline(image_, graph_image_, graph_sub_type_)


    print(image_croping_res.keys())
    print("[INFO] Total time taken to detect the features of graph:", time.time() - s_time)
