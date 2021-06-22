import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


import cv2
from scipy import stats
from scipy import stats
import boto3
# import re

import PIL
from PIL import Image

#Libraries to extract named colors from RGB value
from scipy.spatial import KDTree
import webcolors
from webcolors import CSS3_HEX_TO_NAMES, hex_to_rgb

from statistics import mean
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

import time
import warnings
warnings.filterwarnings("ignore")

# AWS OCR
print("[INFO] AWS Rekognition OCR Model loaded")   
client = boto3.client('rekognition')


# Average value calculation
def Average(lst):
    try:
        avg = sum(lst) / len(lst)
        return int(avg)
    except:
        return 0
    

# To make multilpe word to single line  (for text data of y_axis /x_axis)

def text_allignment(text_list):

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

            if text_list[i][1][1] - text_list[i+1][1][1] in range (-5,5):
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

    # print(text_amend)

    return text_amend

#AWS OCR PIPELINE
def aws_rekognition_ocr(image):
    img_height, img_width, channels = image.shape
    _, im_buf = cv2.imencode('.png', image) # if else
    response = client.detect_text(Image = {"Bytes" : im_buf.tobytes()})

    textDetections = response['TextDetections']

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

            final_text.append((text['DetectedText'],(int(left),int(top),int(right - left),int(bottom - top))))

    return final_text


#  1. Stacked bar with values #################################################################################


#Function to identify the values on y axis ***************************************************************

def y_axis_value_extraction(image_croping_res):
    # rotated_image = image_rotation(image_croping_res['y_axis'][0])
    y_axis_text = aws_rekognition_ocr(image_croping_res['y_axis'][0])
    # print("1st",y_axis_text)
    y_axis_text = text_allignment(y_axis_text)
    # print("2nd",y_axis_text)
    y_axis_data_df = pd.DataFrame(y_axis_text, columns=['char', 'coordinates'])
    y_axis_data_df['coordinates'].tolist()
    pd.DataFrame(y_axis_data_df['coordinates'].tolist(), index=y_axis_data_df.index)
    y_axis_data_df[['left', 'top', 'width', 'height']] = pd.DataFrame(y_axis_data_df['coordinates'].tolist(), index=y_axis_data_df.index)
    y_axis_data_df=y_axis_data_df[['char', 'left', 'top', 'width', 'height']]
    y_axis_data_df = y_axis_data_df.sort_values('top',ignore_index=True)
    # print("y_axis",y_axis_data_df)
    

    return y_axis_data_df



#Function to identify the values on all_bars/bar values***************************************************************
        
def bar_value_extraction(image_croping_res):
    bar_value_text = aws_rekognition_ocr(image_croping_res['all_bars'][0])
    if bar_value_text == []:
        bar_value_df =[]
    else:

    # print("bar_value_text",bar_value_text)
        bar_value_df = pd.DataFrame(bar_value_text, columns=['char', 'coordinates'])
        bar_value_df['coordinates'].tolist()
        pd.DataFrame(bar_value_df['coordinates'].tolist(), index=bar_value_df.index)
        # print("bar_value_df",bar_value_df)
        bar_value_df[['left', 'top', 'width', 'height']] = pd.DataFrame(bar_value_df['coordinates'].tolist(), index=bar_value_df.index)
        bar_value_df=bar_value_df[['char', 'left', 'top', 'width', 'height']]
        bar_value_df = bar_value_df.sort_values('top',ignore_index=True)


    return bar_value_df



def mapping_bar_coordinates_and_bar_values1(bar_value_df):
    bar_value_values = list(bar_value_df["char"])
    bar_value_keys=list(bar_value_df['top'])
    bar_value_left=list(bar_value_df['left'])
    
    return bar_value_keys,bar_value_values,bar_value_left


# Mapping bar values and yaxis data into single data frame
def mapping_yaxis_barvalues_1(y_axis_data_df,bar_value_keys,bar_value_values,bar_value_left):

    model_data = [[i,1] for i in list(bar_value_keys)] #bar xaxis val & i,1 if for 2d araay input for kmeans
    if len(bar_value_keys) == len(y_axis_data_df):
        cluster_length = len(y_axis_data_df)
    elif len(bar_value_keys) < len(y_axis_data_df):
        cluster_length = len(bar_value_keys)
    else:
        cluster_length = len(y_axis_data_df)
    # print("cluster_length",cluster_length)       


    kmeans = KMeans(n_clusters=cluster_length, random_state=0).fit(model_data)
    clustered_bar = [[i, j, k,l] for i, j, k,l in zip(list(bar_value_keys), list(bar_value_values), list(kmeans.labels_),list(bar_value_left))]



    clustered_bar_df = pd.DataFrame(clustered_bar,columns=["char","values","cluster","left"])
    clustered_bar_df.sort_values(['cluster', 'left'], ascending=[True, True], inplace=True)

    # print("clustered_bar_df",clustered_bar_df)

    counter = 1
    clustered_bar_avg_df = pd.DataFrame()
    for each_g in clustered_bar_df.groupby(['cluster']):
        each_g[1]['avg_pos'] = [Average(list(each_g[1]['char']))]*len(each_g[1])
        clustered_bar_avg_df = pd.concat([each_g[1], clustered_bar_avg_df])
        counter = counter + 1
    le = LabelEncoder()
    clustered_bar_avg_df = clustered_bar_avg_df.sort_values(['char'])
    clustered_bar_avg_df['sorted_index'] = le.fit_transform(list(clustered_bar_avg_df['avg_pos']))
    # print("clustered_bar_avg_df",clustered_bar_avg_df)
    y_axis_data_df['sorted_index'] = range(0, len(y_axis_data_df))
    data_df = pd.merge(clustered_bar_avg_df, y_axis_data_df, on="sorted_index")
    data_df ['bar_count']= data_df.groupby('cluster').cumcount().add(1)
    data_df['bar_count'] = 'bar' + data_df['bar_count'].astype(str)
    # print("data_df*********",data_df)
    data_df =data_df[["char_y","bar_count","values"]]
    data_df.drop_duplicates(keep='first',inplace=True)  
    # print(data_df)

    return data_df


# Pipeline for stacked bar with values
def data_extraction_stacked_bar_approach1(image_croping_res):
    y_axis_data_df = y_axis_value_extraction(image_croping_res)
    bar_value_df = bar_value_extraction(image_croping_res)
    bar_value_keys = mapping_bar_coordinates_and_bar_values1(bar_value_df)[0]
    bar_value_values = mapping_bar_coordinates_and_bar_values1(bar_value_df)[1]
    bar_value_left = mapping_bar_coordinates_and_bar_values1(bar_value_df)[2]
    final_df = mapping_yaxis_barvalues_1(y_axis_data_df,bar_value_keys,bar_value_values,bar_value_left)
    print("*********************************************************************")

    # print(final_df)
    return final_df





# 2.STACKED BAR GRAPH WITH NO VALUES#############################################################################################--->


# Get the values and their coordinates of x-axis ***************************************************************************

def x_axis_value_extraction(image_croping_res):
    # rotated_image = image_rotation(image_croping_res['x_axis'][0])
    x_axis_text = aws_rekognition_ocr(image_croping_res['x_axis'][0])
    # print("x_axis_text",x_axis_text)
    x_axis_data_df = pd.DataFrame(x_axis_text, columns=['char', 'coordinates'])
    x_axis_data_df['coordinates'].tolist()
    pd.DataFrame(x_axis_data_df['coordinates'].tolist(), index=x_axis_data_df.index)
    x_axis_data_df[['left', 'top', 'width', 'height']] = pd.DataFrame(x_axis_data_df['coordinates'].tolist(), index=x_axis_data_df.index)
    x_axis_data_df=x_axis_data_df[['char', 'left', 'top', 'width', 'height']]
    x_axis_data_df = x_axis_data_df.sort_values('left',ignore_index=True)


    if (x_axis_data_df['char'][0].find('%') != -1):
        print ("All values in percentage(%) ")
    else:
        print ("All values in Integer")
    x_axis_data_df['char'] = x_axis_data_df['char'].astype(int)
    # print("*************x_axis_data************************")
    # print(x_axis_data_df)

    return x_axis_data_df


#Stage 1 : extract color names using bounding boxes

# extract the RGB values of each bounding boxes


def get_RGB_value(image_croping_res): 
    bar_array = []
    bar = []

    for key in image_croping_res.keys():
        if "bar" in key and key != "all_bars" and "bar_class":
            bar.append(key)
            bar_array.append(image_croping_res[key][0])

    final_xy_list =[]

    for i in range(len(bar_array)):
        x_list=[]
        y_list=[]
        w_list=[]
        h_list=[]
        gray = cv2.cvtColor(bar_array[i], cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray,5)
        binary = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
        inverted_binary = ~binary
        contours, hierarchy = cv2.findContours(inverted_binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        # with_contours = cv2.drawContours(image, contours, -1, (0,255,0), 3)

        for j in contours:

            new_image = bar_array[i].copy()
            f_contour = cv2.drawContours(new_image, contours, 0,(0,255,0),3)
            x, y, w, h = cv2.boundingRect(j)
            cv2.rectangle(f_contour,(x,y), (x+w,y+h), (255,0,0), 5)

            if x>=0 and w>3:
                x_list.append(x)
                y_list.append(y)   
                w_list.append(w)   
                h_list.append(h)   

            # # cv2_imshow(f_contour)

                x_y_dict = {x_list[i]+5: 10 for i in range(len(x_list))}   
                # x=x+5 :y=10 was given manual since system was picking pixel as dividing line between colors
                x_y_dict = sorted(x_y_dict.items())



        final_xy = remove_repeated_contours_xw(x_y_dict)
        final_xy_list1 =[]
        final_xy_list1 = [(k, v) for k, v in final_xy.items()]
        final_xy_list.append(final_xy_list1)



    rgb_pixel_value1=[]
    for j in range(len(bar_array)):
        image_PIL = Image.fromarray(bar_array[j],'RGB')
    #     plt.imshow(image_PIL)
    #     plt.show()
        for k in range(len(final_xy)):
            rgb_pixel_value = image_PIL.getpixel(final_xy_list[j][k])
            rgb_pixel_value1.append(rgb_pixel_value)
#     print(rgb_pixel_value1)

    n = round(len(rgb_pixel_value1) / len(bar_array))
    rgb_pixel_values = [rgb_pixel_value1[i * n:(i + 1) * n] for i in range((len(rgb_pixel_value1) + n - 1) // n )]
    
    return rgb_pixel_values



def convert_rgb_to_names(RGB):
    # a dictionary of all the hex and their respective names in css3
    css3_db = CSS3_HEX_TO_NAMES
    names = []
    rgb_values = []
    for color_hex, color_name in css3_db.items():
        names.append(color_name)
        rgb_values.append(hex_to_rgb(color_hex))
    
    kdt_db = KDTree(rgb_values)
    distance, index = kdt_db.query(RGB)
    return f'{names[index]}'

# Get named colors based on RGB values
def named_color(rgb_pixel_value):
    n_color = []


    for i in range(len(rgb_pixel_value)):
        for j in range (len(rgb_pixel_value[0])):
            n_color.append(convert_rgb_to_names(rgb_pixel_value[i][j]))
#     print(n_color)

    n = len(rgb_pixel_value[0])
    n_colors = [n_color[i * n:(i + 1) * n] for i in range((len(n_color) + n - 1) // n )]

    return n_colors



#Stage 2 : extract values using bounding boxes

# remove repeated contours for x and w values (used to detect stack values)

def remove_repeated_contours_xw(x_w_dict): 
    list1 = []
    final_list = [] 
    for tup in x_w_dict:
        list1.append(tup[0])
    temp = list1[0]
    for index,i in enumerate(list1[1:]):
        if i - temp <= 5:
            list1.pop(index+1)
        else:
            pass
        temp = i   
    for i in list1:
        for y in x_w_dict: 
            if i in y:
                final_list.append(y)
                break
    final = dict(final_list)
    return final


# contour_detection is to create bounding box and identify their values
def contour_detection(image_croping_res):
    
    bar_array = []
    bar = []

    for key in image_croping_res.keys():
        if "bar" in key and key != "all_bars" and "bar_class":
            bar.append(key)
            bar_array.append(image_croping_res[key][0])
    
    final_xw_list =[]
    final_xy_list =[]

    for i in range(len(bar_array)):
        x_list=[]
        y_list=[]
        w_list=[]
        h_list=[]
        gray = cv2.cvtColor(bar_array[i], cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray,5)
        binary = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
        # inverted_binary = ~binary
        contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        # with_contours = cv2.drawContours(image, contours, -1, (0,255,0), 3)

        for j in contours:

            new_image = bar_array[i].copy()
            f_contour = cv2.drawContours(new_image, contours, 0,(0,255,0),3)
            x, y, w, h = cv2.boundingRect(j)
            cv2.rectangle(f_contour,(x,y), (x+w,y+h), (255,0,0), 5)

            if x>=0 and w>3:
                x_list.append(x)
                y_list.append(y)   
                w_list.append(w)   
                h_list.append(h)   

                # plt.imshow(f_contour)
                # plt.show()

                x_w_dict = {x_list[i]: w_list[i] for i in range(len(x_list))}   
                x_w_dict = sorted(x_w_dict.items())
                
        # print("x_w_dict",x_w_dict)
               
        final_xw = remove_repeated_contours_xw(x_w_dict)
        # print("final_xw",final_xw)
        final_xw_list1 =[]
        final_xw_list1 = [(k, v) for k, v in final_xw.items()]
        final_xw_list.append(final_xw_list1)

 
    total_bars = len(bar_array)
    # print("\nfinal_xw_list",final_xw_list)

    return final_xw_list,total_bars


def val_per_pixels(x_axis_data_df):
    total_pixel = max(x_axis_data_df["left"])-min(x_axis_data_df["left"])
    total_value =  max(x_axis_data_df["char"].astype(int))-min(x_axis_data_df["char"].astype(int))
    val_per_pixel = total_value/ total_pixel
    # print("val_per_pixel",val_per_pixel)
    return val_per_pixel 


# stack's X axis value extraction 
def stack_value_extraction(final_xw_list,val_per_pixel,total_bars):

    bar_value=[]
    for i in range(len(final_xw_list)):
        for j in range(len(final_xw_list[0])):
            bar_values = round(final_xw_list[i][j][1] * val_per_pixel)        
            bar_value.append(bar_values)

    n = round(len(bar_value) / total_bars)
    s_bar_value = [bar_value[i * n:(i + 1) * n] for i in range((len(bar_value) + n - 1) // n )]

    stacked_bar = []
    for idx in range(len(s_bar_value)):
        key = "bar_" + str (idx+1)
        stacked_bar.append([key,s_bar_value[idx]])
    df_stacked_val = pd.DataFrame(stacked_bar, columns = ['bar_num', 'values'])
    # print("df_stacked_val",df_stacked_val)

    return df_stacked_val


#mapping y axis value against indivisual bar sequence obtained from graph feature detection
def yaxisval_allbars(image_croping_res,y_axis_data_df):   

    bar_cord = []
    bar1 = []

    for key in image_croping_res.keys():
        if "bar" in key and key != "all_bars" and "bar_class":
            bar1.append(key)
            bar_cord.append(image_croping_res[key][1])

    # print(bar_cord)

    bar2=[]
    for i in range(len(bar_cord)):
        key = "bar_" + str(i+1)
        bar2.append(key)
    # print(bar2)    

    df_yaxis = pd.DataFrame(bar_cord, columns = ['X', 'Y','W','H'])
    df_yaxis["bar_num"] = bar2
    df_yaxis = df_yaxis.sort_values('Y',ignore_index=True)
    df_yaxis["y-axis"] = y_axis_data_df["char"]
    # print("df_yaxis",df_yaxis)
    # print('bar_cord',bar_cord)

    return df_yaxis,bar_cord

def final_dataframe_creation(df_yaxis,df_stacked_val,rgb_pixel_value):
    final_df = pd.merge(df_yaxis, df_stacked_val,on ='bar_num', how ='inner')
    final_df["colors"] = named_color(rgb_pixel_value)
    final_df = final_df[['y-axis','values','colors']]

    return final_df


# Pipeline for stacked bar without values
def data_extraction_stacked_bar_approach2(image_croping_res):
    y_axis_data_df = y_axis_value_extraction(image_croping_res)
    x_axis_data_df = x_axis_value_extraction(image_croping_res)
    rgb_pixel_value = get_RGB_value(image_croping_res)
    n_colors = named_color(rgb_pixel_value)
    final_xw_list = contour_detection(image_croping_res)[0]
    total_bars = contour_detection(image_croping_res)[1]
    val_per_pixel = val_per_pixels(x_axis_data_df)
    df_stacked_val = stack_value_extraction(final_xw_list,val_per_pixel,total_bars)
    df_yaxis = yaxisval_allbars(image_croping_res,y_axis_data_df)[0]
    bar_cord = yaxisval_allbars(image_croping_res,y_axis_data_df)[1]
    final_df = final_dataframe_creation(df_yaxis,df_stacked_val,rgb_pixel_value)
#     print(final_df)
    return final_df

def data_extr_hori_stacked(image_croping_res):
    print("******************************************************************************************")   
    print("[INFO] Data extraction in progress...")
    plt.imshow(image_croping_res['y_axis'][0])
    plt.show()
    plt.imshow(image_croping_res['x_axis'][0])
    plt.show()
    plt.imshow(image_croping_res['all_bars'][0])
    plt.show() 

    data_points = bar_value_extraction(image_croping_res)
    # print("******Data_points_identified_are:****************\n")
    # print("data_points",data_points)

    if str(data_points) == 'None':
        data_point_length = 0
        print("\ndata_point_length:",data_point_length)
    else:
        data_point_length = len(data_points)
        print("\ndata_point_length:",data_point_length)

    # print("\length of data points:",data_point_length)
    if data_point_length > 1:
        print("\n[INFO] STACKED BAR GRAPH WITH VALUES")
        final_df = data_extraction_stacked_bar_approach1(image_croping_res)
    else:
        print("\n[INFO] STACKED BAR GRAPH WITH NO VALUES")
        final_df = data_extraction_stacked_bar_approach2(image_croping_res)

    return final_df

#*****main function for testing*****************************************************************************************

# if __name__ == "__main__":

#     import time
#     from graph_feature_detection import graph_feature_detection_pipeline

#     path = "/home/ideapoke/Desktop/Work/yolo_implementation/Horizonta_graph/stacked_chart/Chart_classification_model_horizontal/images/stacked_with_values.png"
#     # path = "//home/ideapoke/Desktop/Work/yolo_implementation/Horizonta_graph/stacked_chart/Chart_classification_model_horizontal/images/1_stacked_full_image.png"
#     image = cv2.imread(path)
#     image_croping_res = graph_feature_detection_pipeline(image)


#     data_points = char_loc_extraction_val(image_croping_res)
#     print("******Data_points_identified_are:****************\n")
#     print(data_points)
#     print("length of data points:",len(data_points))


    	
#     if len(data_points) > 1:
#     	print("STACKED BAR GRAPH WITH VALUES")
#     	s_time = time.time()
#     	data_extraction_stacked_bar_approach1(image_croping_res)
#     	print(data_extraction_stacked_bar_approach1(image_croping_res))
#     else:
#     	print("STACKED BAR GRAPH WITH NO VALUES")
#     	s_time = time.time()
#     	data_extraction_stacked_bar_approach2(image_croping_res)
#     	print(data_extraction_stacked_bar_approach2(image_croping_res))
#     print("Total time consuption:", time.time() - s_time)