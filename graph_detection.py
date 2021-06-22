# import the necessary packages
import os
import cv2
import time
import numpy as np

import warnings
warnings.filterwarnings("ignore")

# load the COCO class labels our YOLO model was trained on custom dataset
labelsPath = 'cfg/graph_detector.names'
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
                           dtype="uint8")

print("[INFO] Total number of classes", len(COLORS))

# derive the paths to the YOLO weights and model configuration
weightsPath = "weights/graph_detector_6000.weights"

# configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])
configPath = "cfg/graph_detector.cfg"

# load our YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO weight from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
CONFIDENCE_THRESHOLD = 0.3

def model_prediction(image):
    (H, W) = image.shape[:2]
    # determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    # show timing information on YOLO
    print("[INFO] Graph Detection took {:.6f} seconds".format(time.time() - start))

    return layerOutputs, W, H, image


def boundin_box_extraction(layerOutputs, W, H):
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


def image_croping(boxes, confidences, classIDs, image):
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, CONFIDENCE_THRESHOLD)
    all_cropped_images = dict()
    count = 1

    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])

            im_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # crop_img_ = im_rgb[y:y + h, x:x + w]
            crop_img_ = im_rgb
            key_text = text.split(":")[0]
            key_text = key_text + "_" +str(count) # To capture multiple graphs

            all_cropped_images[key_text] = crop_img_
            count = count + 1

    return all_cropped_images


def graph_detection_pipeline(img):
    layerOutputs = model_prediction(img)
    boxes, confidences, classIDs = boundin_box_extraction(layerOutputs[0], layerOutputs[1], layerOutputs[2])
    image = layerOutputs[3]
    all_cropped_images = image_croping(boxes, confidences, classIDs, image)

    return all_cropped_images


if __name__ == "__main__":
    input_file_path = "/home/pushpendu/Documents/All_task/market_related_information/image_data_extraction/data_training_final_2/image/graph_13.jpg"
    img_ = cv2.imread(input_file_path)
    s_time = time.time()
    crop_img = graph_detection_pipeline(img_)
    print("[INFO] Total time taken to detect the graph", time.time() - s_time)
    print(crop_img.keys())

