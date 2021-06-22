import pandas as pd
import numpy as np 
from matplotlib import pyplot as plt
from matplotlib import colors


import cv2
import scipy
from scipy import stats
import boto3



from statistics import mean
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

import binascii
import struct
import scipy.misc
import scipy.cluster

import math
from collections import Counter
import sys 

from scipy.spatial import KDTree
import webcolors
from webcolors import CSS3_HEX_TO_NAMES, hex_to_rgb

from utils import image_rotation

import time
import warnings
warnings.filterwarnings("ignore")

# AWS OCR
print("[INFO] AWS Rekognition OCR Model loaded")   
client = boto3.client('rekognition')


# Outlier detection
def reject_outliers(data):
    d = 1
    m = np.mean(data)
    s = np.std(data)
    filtered = [e for e in data if (m - d * s < e < m + d * s)]
    return filtered

# Average value calculation
def Average(lst):
    try:
        avg = sum(lst) / len(lst)
        return int(avg)
    except:
        return 0

def unique(list1):
    x = np.array(list1)
    unique_val = np.unique(x).tolist()
    
    return unique_val

# To make multilpe word to single line  (for text data of y_axis)

def text_allignment_verticle(text_list):

    text_amend = list()
    l = len(text_list) 
    i = 0
    chars = ""

    bamend = False
    #     print(l)
    while i < l:
        if i+1 < l:
            temp=text_list[i][1][0]
            if text_list[i][1][1] - text_list[i+1][1][1] in range (-5,5):
                if bamend == False:  
                    if temp <= text_list[i+1][1][0]:             
                        chars = text_list[i][0]+ " " +text_list[i+1][0]
                        left=text_list[i][1][0]
                        bottom = text_list[i][1][1]
                        right = text_list[i+1][1][2]
                        top = text_list[i][1][3]
                        bamend = True

                    else:
                        chars = text_list[i+1][0]+ " " +text_list[i][0]
                        left = text_list[i+1][1][0]
                        bottom = text_list[i+1][1][1]
                        right = text_list[i][1][2]
                        top = text_list[i+1][1][3]
                        bamend = True

                else:
                    chars = chars + " " + text_list[i+1][0]
                    bamend = True

            else:

                if bamend == True:
                    text_amend.append((chars,(left,bottom,right,top)))
                    chars=""
                    bamend = False
                else:
                    text_amend.append(text_list[i]) 
        else:
            if chars == "":
                text_amend.append(text_list[i]) # for last record
                chars=""
            else:
                text_amend.append((chars,(left,bottom,right,top)))

        i=i+1

#     print(text_amend)

    return text_amend


# To make multilpe word to single line  (for text data of x_axis)

def text_allignment_horizontal(list1):

#list2 will be the position difference between each character    
    list2=[0]
    for i in range(len(list1)-1):
        list2.append(list1[i+1][1][0] - list1[i][1][2])
#     print(list2)    
# newList will provide the cummulative sum of all the items in list2
    newList = []
    for idx in range(len(list2)-1):

        if idx == 0:
            val = list2[idx] + list2[idx+1]
            newList.append(val)
        else:
            val = list2[idx+1] + newList[idx-1]
            newList.append(val)
    newList.insert(0,0)
#     print(newList)
#test_list will provide all the characters, locations and their position difference
    text_list = list()
    for idx in range(len(list1)):
        val_ele_1 = list(list1[idx])
        dif_ele_1 = newList[idx]
        val_ele_1.append(dif_ele_1)
        text_list.append(val_ele_1)
    text_list   

#     print("text_list",text_list)
# code to put all xaxis words in one line

    text_amend = list()
    l = len(text_list) 
    i = 0
    chars = ""

    bamend = False
    #     print(l)
    while i < l:
        if i+1 < l:
            left=text_list[i][1][0]
    #             print(left)

            if text_list[i][2] - text_list[i+1][2] in range (-5,5):
                if bamend == False:  
                    if left <= text_list[i+1][1][0]:             
                        chars = text_list[i][0]+ " " +text_list[i+1][0]
                        bottom = text_list[i][1][1]
                        right = text_list[i+1][1][2]
                        top = text_list[i][1][3]
                        bamend = True

                    else:
                        chars = text_list[i+1][0]+ " " +text_list[i][0]
                        left = text_list[i+1][1][0]
                        bottom = text_list[i+1][1][1]
                        right = text_list[i][1][2]
                        top = text_list[i+1][1][3]
                        bamend = True

                else:
                    chars = chars + " " + text_list[i+1][0]
                    bamend = True

            else:

                if bamend == True:
                    text_amend.append((chars,(left,bottom,right,top)))
                    chars=""
                    bamend = False
                else:
                    text_amend.append(text_list[i]) 
        else:
            if chars == "":
                text_amend.append(text_list[i]) # for last record
                chars=""
            else:
                text_amend.append((chars,(left,bottom,right,top)))

        i=i+1

#     print("text_amend",text_amend)
    return text_amend


#AWS OCR PIPELINE
def aws_rekognition_ocr(image):
    img_height, img_width, channels = image.shape
    _, im_buf = cv2.imencode('.png', image) # if else
    response = client.detect_text(Image = {"Bytes" : im_buf.tobytes()})

    textDetections = response['TextDetections']
#     print(textDetections)
    final_text=[]
    for text in textDetections:
        
        if text['Type'] == 'WORD' and text['Confidence'] >= 80:
            vertices = [[vertex['X'] * img_width, vertex['Y'] * img_height] for vertex in text['Geometry']['Polygon']]
            vertices = np.array(vertices, np.int32)
            vertices = vertices.reshape((-1, 1, 2))

            left = np.amin(vertices, axis=0)[0][0]
            top = np.amin(vertices, axis=0)[0][1]
            right = np.amax(vertices, axis=0)[0][0]
            bottom = np.amax(vertices, axis=0)[0][1]

#             if left <0 or top < 0 or right < 0 or bottom <0:
#                 pass
            
#             else:
            final_text.append((text['DetectedText'],(int(left),int(top),int(right),int(bottom))))
    
#     if len(final_text) == 0:
#         final_text.append(("no_legend_text",(0,0,0,0)))
    # print("aws final text",final_text)        
    return final_text


def image_enhancement(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))
    dilation = cv2.dilate(thresh1, rect_kernel, iterations = 1)
    # plt.imshow(dilation)
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    im2 = image.copy()
    # text_direct = aws_rekognition_ocr(im2)
    # print("text_direct")
    #option 1: cropping and then extracting image text
    # cropped=[]
    # for cnt in contours:
    #     x, y, w, h = cv2.boundingRect(cnt)
    #     rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #     cropped=(im2[y:y + h, x:x + w]) 
    #     text = aws_rekognition_ocr(cropped)
    #     print(text)

    #option 2: using bounding box and then extracting image text
    image_text1=[]
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt) 
        rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 255), 2) 
        # plt.imshow(rect)
        text1 = aws_rekognition_ocr(rect)
    image_text1.append(text1)
      
    image_text = [item for sublist in image_text1 for item in sublist] # list of list to list
    # print("image_enhancement_text",image_text)
  
    return image_text
# 1. GROUPED BAR GRAPH WITH VALUES #############################################################################################--->


#Function to identify the values on x axis ***************************************************************

# Get the values and their coordinates of x-axis ***************************************************************************

def x_axis_value_extraction(image_croping_res):
    # rotated_image = image_rotation(image_croping_res['x_axis'][0])
    # plt.imshow(image_croping_res['x_axis'][0])
    # plt.show()
    x_axis_text_aws = aws_rekognition_ocr(image_croping_res['x_axis'][0])
    # print("aws",x_axis_text_aws)
    x_axis_text = image_enhancement(image_croping_res['x_axis'][0])
    # print("x_axis_text",x_axis_text)   
    x_axis_text1 = text_allignment_horizontal(x_axis_text)
    # print("allignment",x_axis_text1)  
    x_axis_data_df = pd.DataFrame(x_axis_text1, columns=['char', 'coordinates','others'])
    # print(x_axis_data_df)
    x_axis_data_df['coordinates'].tolist()
    pd.DataFrame(x_axis_data_df['coordinates'].tolist(), index=x_axis_data_df.index)
    x_axis_data_df[['left', 'top', 'right', 'bottom']] = pd.DataFrame(x_axis_data_df['coordinates'].tolist(), index=x_axis_data_df.index)
    x_axis_data_df=x_axis_data_df[['char', 'left', 'top', 'right', 'bottom']]
    x_axis_data_df = x_axis_data_df.sort_values('left',ignore_index=True)

    # x_axis_data_df['char'] = x_axis_data_df['char'].astype(int)
    # print("*************x_axis_data************************")
    # print(x_axis_data_df)

    return x_axis_data_df


#Function to identify the values on all_bars/bar values***************************************************************

def bar_value_extraction(image_croping_res):
    bar_value_text = aws_rekognition_ocr(image_croping_res['all_bars'][0])
    if bar_value_text == [] or len(bar_value_text)<=2:
        bar_value_df =[]
    else:

    # print("bar_value_text",bar_value_text)
        bar_value_df = pd.DataFrame(bar_value_text, columns=['char', 'coordinates'])
        bar_value_df['coordinates'].tolist()
        pd.DataFrame(bar_value_df['coordinates'].tolist(), index=bar_value_df.index)
        # print("bar_value_df",bar_value_df)
        bar_value_df[['left', 'top', 'width', 'height']] = pd.DataFrame(bar_value_df['coordinates'].tolist(), index=bar_value_df.index)
        bar_value_df=bar_value_df[['char', 'left', 'top', 'width', 'height']]
        bar_value_df = bar_value_df.sort_values('left',ignore_index=True)
        # print(bar_value_df)

    return bar_value_df

def mapping_bar_coordinates_and_bar_values1(bar_value_df):
    bar_value_values = list(bar_value_df["char"])
    bar_value_keys=list(bar_value_df['left'])

    
    return bar_value_keys,bar_value_values


#******************Legend color identification and mapping******************

class BackgroundColorDetectors():
    def __init__(self, imageLoc):
#         self.img = cv2.imread(imageLoc, 1)
        self.img = imageLoc
#         print(self.img)
        self.manual_count = {}
        self.w, self.h, self.channels = self.img.shape
        self.total_pixels = self.w*self.h

    def count(self):
        for y in range(0, self.h):
            for x in range(0, self.w):
                RGB = (self.img[x, y, 2], self.img[x, y, 1], self.img[x, y, 0])
                if RGB in self.manual_count:
                    self.manual_count[RGB] += 1
                else:
                    self.manual_count[RGB] = 1

    def average_colour(self):
        red = 0
        green = 0
        blue = 0
        sample = 10
        for top in range(0, sample):
            red += self.number_counter[top][0][0]
            green += self.number_counter[top][0][1]
            blue += self.number_counter[top][0][2]

        average_red = red / sample
        average_green = green / sample
        average_blue = blue / sample
#         print("Average RGB for top ten is: (", average_red,
#               ", ", average_green, ", ", average_blue, ")")

    def twenty_most_common(self):
        self.count()
        self.number_counter = Counter(self.manual_count).most_common(20)
        # for rgb, value in self.number_counter:
        #     print(rgb, value, ((float(value)/self.total_pixels)*100))

    def detect(self):
        self.twenty_most_common()
        self.percentage_of_first = (
            float(self.number_counter[0][1])/self.total_pixels)
        self.percentage_of_first
        if self.percentage_of_first > 0.5:
            self.number_counter[0][0]
#             print("Background color is ", self.number_counter[0][0])
        else:
            self.average_colour()
            
        return self.number_counter[0][0]

def background_rgb(image_croping_res):
    if 'legend' in image_croping_res:
        background_image = image_croping_res["legend"][0]
        BackgroundColor = BackgroundColorDetectors(background_image)
        background = BackgroundColor.detect()
        # print("background",background)
    else:
        background = (255, 255, 255)

    return background 

def identify_legend(image_croping_res):

    legend_color_array = []
    legend_color = []
    legend_color_loc =[]

    for key in image_croping_res.keys():

        if "legend_color" in key:
            legend_color.append(key)
            legend_color_array.append(image_croping_res[key][0])
            legend_color_loc.append(image_croping_res[key][1])

    for i in range(len(legend_color_loc)-1):
        if legend_color_loc[i+1][0] - legend_color_loc[i][0] in range(-15,15):
            type_legend = "verticle"
        else:
            type_legend = "horizontal"

    print("type_legend",type_legend)
    
    return  type_legend  



def closest_colour(requested_colour):
    min_colours = {}
    for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_colour[0]) ** 2
        gd = (g_c - requested_colour[1]) ** 2
        bd = (b_c - requested_colour[2]) ** 2
        min_colours[(rd + gd + bd)] = name
    return min_colours[min(min_colours.keys())]

def pythagoras_distance(point_1, point_2):
    x1 = point_1[0]
    y1 = point_1[1]
    z1 = point_1[2]
    
    x2 = point_2[0]
    y2 = point_2[1]
    z2 = point_2[2]
    
    dist = math.sqrt(math.pow(x2 - x1, 2) + math.pow(y2 - y1, 2) + math.pow(z2 - z1, 2))
    
    return round(dist,2)


def furthest_color_detection(colors,background):
    dis_color = {pythagoras_distance(color, background): color for color in colors}
    furthest_color = dis_color[max(dis_color.keys())]
    
    return furthest_color

def get_colour_name(requested_colour):
    try:
        closest_name = actual_name = webcolors.rgb_to_name(requested_colour)
    except ValueError:
        closest_name = closest_colour(requested_colour)
        actual_name = None
    return closest_name

def named_colors(rgb_values):
    named_color=[]
    for i in range(len(rgb_values)):
        named_color.append(["bar"+str(i+1), get_colour_name(rgb_values[i])])
    named_color = pd.DataFrame(named_color,columns=["bar_num","bar_color"])
#     print(named_color)
    return named_color 

def get_RGB_value_legend(image_array,background): 
    legend_colors=[]
    for j in range(len(image_array)):
        NUM_CLUSTERS = 3
        im = image_array[j]
        ar = np.asarray(im)
        shape = ar.shape
        #         print(shape)
        ar = ar.reshape(scipy.product(shape[:2]), shape[2]).astype(float)

        #     print('finding clusters')
        codes, dist = scipy.cluster.vq.kmeans(ar, NUM_CLUSTERS)
        # print('cluster centres:\n', codes)

        codes_new=[]
        for k in range(len(codes)):
            codes_new.append(codes[k].tolist())

        codes_newest=[]
        for l in range(len(codes_new)):
            codes_newest.append([int(item) for item in codes_new[l]])
#         print(codes_newest)

        
        legend_values = furthest_color_detection(codes_newest,background)
        legend_colors.append(legend_values)
    # print("legend_colors",legend_colors)
    return legend_colors

def legend_color(image_croping_res,background,type_legend): 
    # plt.imshow(image_croping_res['legend'][0])
    # plt.show()
    legend_color_array = []
    legend_color = []
    legend_color_loc =[]

    for key in image_croping_res.keys():

        if "legend_color" in key:
            legend_color.append(key)
            legend_color_array.append(image_croping_res[key][0])

            # legend_color_array.append(cv2.cvtColor(image_croping_res[key][0],cv2.COLOR_BGR2RGB))
#             legend_color_array.append(image_croping_res[key][0])
            legend_color_loc.append(image_croping_res[key][1])
            # plt.imshow(image_croping_res[key][0])
            # plt.show()
    # print(len(legend_color_array))
    
    legend_color_values = get_RGB_value_legend(legend_color_array,background)
    # print("legend_color_values",legend_color_values)
    legend_colors_df = named_colors(legend_color_values)
    legend_colors_df["coordinates"] = legend_color_loc
    legend_colors_df['coordinates'].tolist()
    pd.DataFrame(legend_colors_df['coordinates'].tolist(), index=legend_colors_df.index)
    # print("legend_colors_df",legend_colors_df)
    legend_colors_df[['left', 'top', 'right', 'bottom']] = pd.DataFrame(legend_colors_df['coordinates'].tolist(), index=legend_colors_df.index)
    
    if type_legend == 'verticle':
        legend_colors_df.sort_values('top', ascending=True, inplace=True)
    else:
        legend_colors_df.sort_values('left', ascending=True, inplace=True)  

    # print("legend_colors_df",legend_colors_df)
    
    return legend_colors_df

def legend_text(image_croping_res,type_legend): 

    legend_text_array = []
    legend_text = []
    legend_color_loc=[]
    for key in image_croping_res.keys():
        if "legend_text" in key:
            legend_text.append(key)
            legend_text_array.append(image_croping_res[key][0])
            legend_color_loc.append(image_croping_res[key][1])
    # print("legend_color_loc",legend_color_loc)
    legend_text_val1=[]
    for i in range(len(legend_text_array)):
        legend_text_val1.append(image_enhancement(legend_text_array[i]))
        
    print("legend_text_val1",legend_text_val1)
    legend_text_val = [item for sublist in legend_text_val1 for item in sublist]
    print("legend_text_val2",legend_text_val)

    if type_legend == 'verticle':
        legend_text_val = text_allignment_verticle(legend_text_val)
        # print("verti")
    else: 
        legend_text_val = text_allignment_horizontal(legend_text_val)
        # print("hori")
    # print("new_legend_text_val3",legend_text_val)
    # print("text",legend_text_val[2][0][0][0]) 
    # print("coordinates",legend_text_val[0][1])

    column1 =[]
    column2 =[]
    for j in range(len(legend_text_val)):
        column1.append(legend_text_val[j][0])
        column2.append(legend_color_loc[j])
    legend_text_df = pd.DataFrame(list(zip(column1, column2)),columns =['text', 'coordinates'])
    print("legend_text_df",legend_text_df)

    legend_text_df['coordinates'].tolist()
    pd.DataFrame(legend_text_df['coordinates'].tolist(), index=legend_text_df.index)
    legend_text_df[['left', 'top', 'right', 'bottom']] = pd.DataFrame(legend_text_df['coordinates'].tolist(), index=legend_text_df.index)
    
    if type_legend == 'verticle':
        legend_text_df.sort_values('top', ascending=True, inplace=True)
    else:
        legend_text_df.sort_values('left', ascending=True, inplace=True)
    # print("legend_text_df",legend_text_df)
    
    return legend_text_df

def mapping_legend_color_text(legend_text_df,legend_colors_df,type_legend):
    # print("legend_text_df",legend_text_df)
    # print("legend_colors_df",legend_colors_df)
    if type_legend == 'verticle':
        pos = 'top'
    else:
        pos= 'left'
    legend_value_keys = legend_text_df[pos]
    legend_value_values = legend_text_df['text']

    cluster_length = len(legend_text_df)
    model_data = [[i,1] for i in list(legend_value_keys)]

    kmeans = KMeans(n_clusters=cluster_length, random_state=0).fit(model_data)
    legend_stack = [[i,j,k] for i,j,k in zip(list(legend_value_keys), list(legend_value_values), list(kmeans.labels_))]
    # print("legend_stack",legend_stack)
    legend_stack_df = pd.DataFrame(legend_stack,columns=[pos,"char","cluster"])
    # clustered_stack_df.sort_values(['cluster', 'top'], ascending=[True, True], inplace=True)
    counter = 1
    legend_stack_avg_df = pd.DataFrame()
    for each_g in legend_stack_df.groupby(["cluster"]):
        each_g[1]['avg_pos'] = [Average(list(each_g[1][pos]))]*len(each_g[1])
        legend_stack_avg_df = pd.concat([each_g[1], legend_stack_avg_df])
        counter = counter + 1
    le = LabelEncoder()
    # print("legend_stack_avg_df_before sort",legend_stack_avg_df)
    legend_stack_avg_df = legend_stack_avg_df.sort_values([pos])
    legend_stack_avg_df['sorted_index'] = le.fit_transform(list(legend_stack_avg_df['avg_pos']))
    # print("legend_stack_avg_df_after sort",legend_stack_avg_df)
    legend_colors_df['sorted_index'] = range(0, len(legend_colors_df))
    legend_df = pd.merge(legend_stack_avg_df, legend_colors_df, on="sorted_index")
    legend_df =legend_df[["char","bar_color"]]
    legend_df.rename(columns = {'bar_color':'L_color'}, inplace = True)
    print("legend_df",legend_df)    

    return legend_df

# Mapping bar values and yaxis data into single data frame
def mapping_legend_and_data_1(x_axis_data_df,bar_value_keys,bar_value_values,legend_df):

    model_data = [[i,1] for i in list(bar_value_keys)] #bar xaxis val & i,1 if for 2d araay input for kmeans
    if len(bar_value_keys) == len(x_axis_data_df):
        cluster_length = len(x_axis_data_df)
    elif len(bar_value_keys) < len(x_axis_data_df):
        cluster_length = len(bar_value_keys)
    else:
        cluster_length = len(x_axis_data_df)
    # print("cluster_length",cluster_length)       


    kmeans = KMeans(n_clusters=cluster_length, random_state=0).fit(model_data)
    clustered_bar = [[i, j, k] for i, j, k in zip(list(bar_value_keys), list(bar_value_values), list(kmeans.labels_))]



    clustered_bar_df = pd.DataFrame(clustered_bar,columns=["char","values","cluster"])
    

    counter = 1
    clustered_bar_avg_df = pd.DataFrame()
    for each_g in clustered_bar_df.groupby(['cluster']):
        each_g[1]['avg_pos'] = [Average(list(each_g[1]['char']))]*len(each_g[1])
        clustered_bar_avg_df = pd.concat([each_g[1], clustered_bar_avg_df])
        counter = counter + 1
    le = LabelEncoder()
    clustered_bar_avg_df = clustered_bar_avg_df.sort_values(['char'])
    # print("clustered_bar_df",clustered_bar_avg_df)
    clustered_bar_avg_df['sorted_index'] = le.fit_transform(list(clustered_bar_avg_df['avg_pos']))
    # print("clustered_bar_avg_df",clustered_bar_avg_df)
    x_axis_data_df['sorted_index'] = range(0, len(x_axis_data_df))
    data_df = pd.merge(clustered_bar_avg_df, x_axis_data_df, on="sorted_index")
    data_df ['bar_count']= data_df.groupby('cluster').cumcount().add(1)
    data_df['bar_count'] = 'bar' + data_df['bar_count'].astype(str)
    # print("data_df*********",data_df)
    data_df =data_df[["char_y","bar_count","values"]]
    data_df.rename(columns = {'char_y':'x_axis_values','values':'Bar_values'}, inplace = True)
    data_df.drop_duplicates(keep='first',inplace=True)  
    # print("data_df1*********",data_df)
    bar_list=[]
    for i in range(len(legend_df)):
        bar_list.append("bar" + str(i+1))
#     print(bar_list)
    legend_df["bar_count"]=bar_list
    # print(legend_df)
    data_df = pd.merge(data_df,legend_df,on ='bar_count',how ='left')
    data_df.drop('L_color', inplace=True, axis=1)

    # data_df.rename(columns = {'char_y':'char_x'}, inplace = True)
    # data_df = data_df[["char_x","char","L_color","values"]]

    return data_df

def data_extraction_grouped_bar_approach1(image_croping_res):
    x_axis_data_df = x_axis_value_extraction(image_croping_res)
    bar_value_df = bar_value_extraction(image_croping_res)
    bar_value_keys = mapping_bar_coordinates_and_bar_values1(bar_value_df)[0]
    bar_value_values = mapping_bar_coordinates_and_bar_values1(bar_value_df)[1]
    
    background= background_rgb(image_croping_res)
    type_legend = identify_legend(image_croping_res)
    legend_colors_df = legend_color(image_croping_res,background,type_legend)
    legend_text_df = legend_text(image_croping_res,type_legend)
    legend_df=mapping_legend_color_text(legend_text_df,legend_colors_df,type_legend)
   
    final_df = mapping_legend_and_data_1(x_axis_data_df,bar_value_keys,bar_value_values,legend_df)
    print("*********************************************************************")

    # print(final_df)
    return final_df


# 2.GROUPED BAR GRAPH WITH NO VALUES#############################################################################################--->


# Get the values and their coordinates of y_axis ******************************************************************

def y_axis_value_extraction(image_croping_res):
    # rotated_image = image_rotation(image_croping_res['y_axis'][0])
    y_axis_text = aws_rekognition_ocr(image_croping_res['y_axis'][0])
    # print("1st",y_axis_text)
    y_axis_text = text_allignment_verticle(y_axis_text)
    # print("2nd",y_axis_text)
    y_axis_data_df = pd.DataFrame(y_axis_text, columns=['char', 'coordinates'])
    y_axis_data_df['coordinates'].tolist()
    pd.DataFrame(y_axis_data_df['coordinates'].tolist(), index=y_axis_data_df.index)
    y_axis_data_df[['left', 'top', 'width', 'height']] = pd.DataFrame(y_axis_data_df['coordinates'].tolist(), index=y_axis_data_df.index)
    y_axis_data_df=y_axis_data_df[['char', 'left', 'top', 'width', 'height']]
    y_axis_data_df = y_axis_data_df.sort_values('top',ignore_index=True)
    # print("y_axis",y_axis_data_df)
    

    return y_axis_data_df


# Identify the coordinates of the bar such that it can be mapped with y_axis********************************************
        
def bar_coordinates_detection(image_croping_res):
    bar_pos = []
    bar = []

    for key in image_croping_res.keys():
        if "bar" in key and key != "all_bars" and key != "bar_class":
            bar.append(key)
            bar_pos.append(image_croping_res[key][1])

    # print(bar)       
    # print(bar_pos)

    # print("*** bar_pos ***", bar_pos)
    bar_coordinates = pd.DataFrame(bar_pos, columns=['x', 'y', 'w', 'h'], index=bar)
    bar_coordinates['bars'] = bar_coordinates.index
    bar_coordinates.reset_index(drop=True, inplace=True)
    bar_coordinates = bar_coordinates.sort_values('x',ascending=True)

    # print("*************bar_coordinates************************")
    # print(bar_coordinates)
    return bar_coordinates
 

#Calculate bar value by mapping bar coordinates to X-axis oordinates*******************************************************
def bar_value_calculation(y_axis_data_df, bar_coordinates):
    dif_h = []
    dif_v = []
    dif_h1=[]
    bar_cal_values = []
    for idx in range(len(y_axis_data_df) - 1):
        val_1 = y_axis_data_df.loc[idx]["top"]
        val_2 = y_axis_data_df.loc[idx + 1]["top"]
        dif_h1.append(val_2 - val_1)
    # print("dif", dif_h1)
    dif_h = reject_outliers(dif_h1)
    if dif_h == []:
        dif_h = [min(dif_h1)]
    else:
        dif_h = dif_h

    # print("dif_h",dif_h)

    if len(dif_h)>1:
        avg_dif = Average(dif_h)
    else:
        avg_dif = dif_h
    # print("avg_dif",avg_dif)


    for idx in range(len(y_axis_data_df) - 1):
        val_2 = float(y_axis_data_df.loc[idx]["char"].replace('%', ''))
        val_1 = float(y_axis_data_df.loc[idx + 1]["char"].replace('%', ''))

        dif_v.append(val_2 - val_1)


    dif_val = stats.mode(dif_v)[0][0]
    # print("dif_val",dif_val)

    for i in list(bar_coordinates['h']):
        bar_cal_values.append((dif_val / avg_dif) * i,)

    bar_cal_values =list(map(int, bar_cal_values))
    # print("*************bar_cal_values**************")
    # print(bar_cal_values)
    return bar_cal_values

# extract the RGB values from Bar
def all_bar_array(image_croping_res): 
    bar_array = []
    bar = []

    for key in image_croping_res.keys():
        if "bar" in key and key != "all_bars" and key != "bar_class":
            bar.append(key)
            bar_array.append(image_croping_res[key][0])
            # bar_array.append(cv2.cvtColor(image_croping_res[key][0],cv2.COLOR_BGR2RGB))
    return bar_array

def get_RGB_value_bar(bar_array): 


    rgb_value=[]
    for j in range(len(bar_array)):
        NUM_CLUSTERS = 5
        im = bar_array[j]
        ar = np.asarray(im)
        shape = ar.shape
        ar = ar.reshape(scipy.product(shape[:2]), shape[2]).astype(float)

    #     print('finding clusters')
        codes, dist = scipy.cluster.vq.kmeans(ar, NUM_CLUSTERS)
    #     print('cluster centres:\n', codes)

        vecs, dist = scipy.cluster.vq.vq(ar, codes)         # assign codes
        counts, bins = scipy.histogram(vecs, len(codes))    # count occurrences

        index_max = scipy.argmax(counts)                    # find most frequent
        peak = codes[index_max]
        colour = binascii.hexlify(bytearray(int(c) for c in peak)).decode('ascii')
    #     print('most frequent is %s (#%s)' % (peak, colour))
        rgb_value.append(peak)
    rgb_values1=[]
    for k in range(len(rgb_value)):
        rgb_values1.append(rgb_value[k].tolist())
    rgb_values=[]
    for l in range(len(rgb_values1)):
        rgb_values.append([int(item) for item in rgb_values1[l]])
    #     rgb_values.append("stack")


#     print(rgb_values)
    return rgb_values




def mappig_bar_x_and_bar_values_2(bar_coordinates,bar_cal_values):
    bar_value_values= bar_cal_values
    bar_value_keys=list(bar_coordinates['x'])
    bar_num = list(bar_coordinates['bars'])
    # print("bar_values***:",bar_values)
    return bar_value_keys,bar_value_values,bar_num


def mapping_xaxis_barvalues_2(x_axis_data_df,bar_value_keys,bar_value_values,bar_num,bar_named_color):

    model_data = [[i,1] for i in list(bar_value_keys)] #bar xaxis val & i,1 if for 2d araay input for kmeans

    kmeans = KMeans(n_clusters=len(x_axis_data_df), random_state=0).fit(model_data)
    clustered_bar = [[i, j, k,l] for i, j, k,l in zip(list(bar_value_keys), list(bar_value_values), list(kmeans.labels_),list(bar_num))]

    clustered_bar_df = pd.DataFrame(clustered_bar,columns=["x_pos","values","cluster","bar_num"])
    # print("clustered_bar_df",clustered_bar_df)

    counter = 1
    clustered_bar_avg_df = pd.DataFrame()
    for each_g in clustered_bar_df.groupby(['cluster']):
        each_g[1]['avg_pos'] = [Average(list(each_g[1]['x_pos']))]*len(each_g[1])
        clustered_bar_avg_df = pd.concat([each_g[1], clustered_bar_avg_df])
        counter = counter + 1
    le = LabelEncoder()
    clustered_bar_avg_df = clustered_bar_avg_df.sort_values(['x_pos'])
    clustered_bar_avg_df['sorted_index'] = le.fit_transform(list(clustered_bar_avg_df['avg_pos']))
    # print("clustered_bar_avg_df",clustered_bar_avg_df)
    x_axis_data_df['sorted_index'] = range(0, len(x_axis_data_df))

    data_df = pd.merge(clustered_bar_avg_df, x_axis_data_df, on="sorted_index")
    data_df ['bar_count']= data_df.groupby('cluster').cumcount().add(1)
    data_df['bar_count'] = 'bar' + data_df['bar_count'].astype(str)

    # print("data_df*********",data_df)
    data_df =data_df[["char","bar_count","values","bar_num"]]
    data_df.rename(columns = {'char':'x_axis_values','values':'Bar_values'}, inplace = True)
    data_df.drop_duplicates(keep='first',inplace=True)  
    data_df = pd.merge(data_df, bar_named_color, on="bar_num")
    # print(data_df)

    return data_df

def named_color_to_rgb(uniquelist):
    rgb_col_val=[]
    for j in range(len(uniquelist)):
        rgb_col_val.append(colors.hex2color(colors.cnames[uniquelist[j]]))
#     print(rgb_col_val)
    lists=[]
    for i in range(len(rgb_col_val)):
        lists.append([round(rgb_col_val[i][0]*255),round(rgb_col_val[i][1]*255),round(rgb_col_val[i][2]*255)])

    return lists

def nearest_color_detection(colors,test):
    dis_color = {pythagoras_distance(color, test): color for color in colors}
    nearest_color = dis_color[min(dis_color.keys())]
    
    return nearest_color

def mapping_legend_and_data_2(legend_df,data_df):
    legendcol = list(legend_df["L_color"])
    barcol= list(data_df["bar_color"])
    u_legendcol = unique(legendcol)
    u_barcol = unique(barcol)

    n_legendcol = named_color_to_rgb(u_legendcol)
    n_barcol= named_color_to_rgb(u_barcol)


    df2 = pd.DataFrame(list(zip(u_barcol,n_barcol)),columns =['B_Name','B_RGB'])
    result=[]
    for i in range(len(n_barcol)):
        test= n_barcol[i]
        result.append(nearest_color_detection(n_legendcol,test))
#     print(result) 
    color_result = named_colors(result)
    bar_color = color_result["bar_color"].tolist()
    df2["near_rgb"] =result
    df2["bar_color"] =bar_color
    df1 = pd.DataFrame(list(zip(u_legendcol, n_legendcol)),columns =['bar_color', 'L_RGB'])
    color_result = pd.merge(df2,df1,on ='bar_color',how ='left')
    color_result = color_result[['B_Name','bar_color']]
    color_result.rename(columns = {'bar_color':'L_color','B_Name':'bar_color'}, inplace = True)
    data_df_new = pd.merge(data_df,color_result,on ='bar_color',how ='left')
    final_df = pd.merge(data_df_new,legend_df,on ='L_color',how ='left')
    final_df = final_df[["x_axis_values","char","bar_color","Bar_values"]]
    final_df.rename(columns = {'x_axis_values':'x_axis','char':'category'}, inplace = True)
    # final_df.sort_values(['x_axis', 'category'], ascending=[True, True],inplace=True)
    # print("final_df",final_df)
    return final_df

#Final pipeline code ***************************************************************************************************

def data_extraction_grouped_bar_approach2(image_croping_res):
    type_legend = identify_legend(image_croping_res)

    x_axis_data_df = x_axis_value_extraction(image_croping_res)
    y_axis_data_df = y_axis_value_extraction(image_croping_res)
    bar_array = all_bar_array(image_croping_res)
    bar_rgb_values = get_RGB_value_bar(bar_array)
    bar_named_color = named_colors(bar_rgb_values)
    background= background_rgb(image_croping_res)
    type_legend = identify_legend(image_croping_res)
    legend_colors_df = legend_color(image_croping_res,background,type_legend)
    legend_text_df = legend_text(image_croping_res,type_legend)
    legend_df=mapping_legend_color_text(legend_text_df,legend_colors_df,type_legend)
    bar_coordinates = bar_coordinates_detection(image_croping_res)
    bar_cal_values = bar_value_calculation(y_axis_data_df, bar_coordinates)

    bar_value_keys = mappig_bar_x_and_bar_values_2(bar_coordinates,bar_cal_values)[0]
    bar_value_values = mappig_bar_x_and_bar_values_2(bar_coordinates,bar_cal_values)[1]
    bar_num = mappig_bar_x_and_bar_values_2(bar_coordinates,bar_cal_values)[2]
    bar_values = mappig_bar_x_and_bar_values_2(bar_coordinates,bar_cal_values)
    data_df = mapping_xaxis_barvalues_2(x_axis_data_df,bar_value_keys,bar_value_values,bar_num,bar_named_color)
    final_df = mapping_legend_and_data_2(legend_df,data_df)

    print("*********************************************************************")

    # print(final_df)
    return final_df

def data_ext_verti_grouped(image_croping_res):

    print("******************************************************************************************")   
    print("[INFO] Data extraction in progress...")
    # plt.imshow(image_croping_res['y_axis'][0])
    # plt.show()
    plt.imshow(image_croping_res['x_axis'][0])
    plt.show()
    # plt.imshow(image_croping_res['all_bars'][0])
    # plt.show() )
    data_points = bar_value_extraction(image_croping_res)
    print("******Data_points_identified_are:****************\n")
    print("data_points",data_points)

    if str(data_points) == 'None':
        data_point_length = 0
        print("\ndata_point_length:",data_point_length)
    else:
        data_point_length = len(data_points)
        print("\ndata_point_length:",data_point_length)

    # print("\length of data points:",data_point_length)
    if data_point_length > 1:
        print("[INFO] GROUPED BAR GRAPH WITH VALUES")
        final_df = data_extraction_grouped_bar_approach1(image_croping_res)
    else:
        print("[INFO] GROUPED BAR GRAPH WITH NO VALUES")
        final_df = data_extraction_grouped_bar_approach2(image_croping_res)

    return final_df

# if __name__ == "__main__":

#     import time
#     from graph_feature_detection import graph_feature_detection_pipeline
    
#     path = "/home/ideapoke/Desktop/Work/yolo_implementation/verticle_bar_graph/grouped/images/without_val_1.png"
    
#     image = cv2.imread(path)
#     graph_image = "Vertical Bar Graph"
#     graph_sub_type = "grouped_bar"
#     image_croping_res = graph_feature_detection_pipeline(image, graph_image, graph_sub_type)
    
#     print(data_ext_verti_grouped(image_croping_res))