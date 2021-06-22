import pandas as pd
import numpy as np 
from matplotlib import pyplot as plt


import cv2
import scipy
from scipy import stats
import boto3



from statistics import mean
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

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

# 1. GROUPED BAR GRAPH WITH VALUES #############################################################################################--->


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
        # print(bar_value_df)

    return bar_value_df


def mapping_bar_coordinates_and_bar_values1(bar_value_df):
    bar_value_values = list(bar_value_df["char"])
    bar_value_keys=list(bar_value_df['top'])

    
    return bar_value_keys,bar_value_values


# Mapping bar values and yaxis data into single data frame
def mapping_yaxis_barvalues_1(y_axis_data_df,bar_value_keys,bar_value_values):

    model_data = [[i,1] for i in list(bar_value_keys)] #bar xaxis val & i,1 if for 2d araay input for kmeans
    if len(bar_value_keys) == len(y_axis_data_df):
        cluster_length = len(y_axis_data_df)
    elif len(bar_value_keys) < len(y_axis_data_df):
        cluster_length = len(bar_value_keys)
    else:
        cluster_length = len(y_axis_data_df)
    # print("cluster_length",cluster_length)       


    kmeans = KMeans(n_clusters=cluster_length, random_state=0).fit(model_data)
    clustered_bar = [[i, j, k] for i, j, k in zip(list(bar_value_keys), list(bar_value_values), list(kmeans.labels_))]



    clustered_bar_df = pd.DataFrame(clustered_bar,columns=["char","values","cluster"])
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


#Final pipeline code ***************************************************************************************************

def data_extraction_grouped_bar_approach1(image_croping_res):
    y_axis_data_df = y_axis_value_extraction(image_croping_res)
    bar_value_df = bar_value_extraction(image_croping_res)
    bar_value_keys = mapping_bar_coordinates_and_bar_values1(bar_value_df)[0]
    bar_value_values = mapping_bar_coordinates_and_bar_values1(bar_value_df)[1]
    final_df = mapping_yaxis_barvalues_1(y_axis_data_df,bar_value_keys,bar_value_values)
    print("*********************************************************************")

    # print(final_df)
    return final_df



# 2.GROUPED BAR GRAPH WITH NO VALUES#############################################################################################--->


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
    # print("*************x_axis_data************************")
    # print(x_axis_data_df)

    return x_axis_data_df


# identify the coordinates of the bar such that it can be mapped with X-axis********************************************
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
    bar_coordinates = bar_coordinates.sort_values('y',ascending=True)

    # print("*************bar_coordinates************************")
    # print(bar_coordinates)
    return bar_coordinates


#Calculate bar value by mapping bar coordinates to X-axis oordinates*******************************************************
def bar_value_calculation(x_axis_data_df, bar_coordinates):
    dif_h = []
    dif_v = []
    dif_h1=[]
    bar_cal_values = []
    for idx in range(len(x_axis_data_df) - 1):
        val_1 = x_axis_data_df.loc[idx]["left"]
        val_2 = x_axis_data_df.loc[idx + 1]["left"]
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


    for idx in range(len(x_axis_data_df) - 1):
        val_1 = float(x_axis_data_df.loc[idx]["char"].replace('%', ''))
        val_2 = float(x_axis_data_df.loc[idx + 1]["char"].replace('%', ''))

        dif_v.append(val_2 - val_1)


    dif_val = stats.mode(dif_v)[0][0]
    # print("dif_val",dif_val)

    for i in list(bar_coordinates['w']):
        bar_cal_values.append((dif_val / avg_dif) * i,)

    bar_cal_values =list(map(int, bar_cal_values))
    # print("*************bar_cal_values**************")
    # print(bar_cal_values)
    return bar_cal_values


def mappig_bar_y_and_bar_values_2(bar_coordinates,bar_cal_values):
    list1= bar_cal_values
    list2=list(bar_coordinates['y'])

    bar_values = dict(zip(list2, list1))
    # print("bar_values***:",bar_values)
    return bar_values



def mapping_yaxis_barvalues_2(y_axis_data_df,bar_values):

    model_data = [[i,1] for i in list(bar_values.keys())] #bar xaxis val & i,1 if for 2d araay input for kmeans

    kmeans = KMeans(n_clusters=len(y_axis_data_df), random_state=0).fit(model_data)
    clustered_bar = [[i, j, k] for i, j, k in zip(list(bar_values.keys()), list(bar_values.values()), list(kmeans.labels_))]



    clustered_bar_df = pd.DataFrame(clustered_bar,columns=["y_pos","values","cluster"])
    # print("clustered_bar_df",clustered_bar_df)

    counter = 1
    clustered_bar_avg_df = pd.DataFrame()
    for each_g in clustered_bar_df.groupby(['cluster']):
        each_g[1]['avg_pos'] = [Average(list(each_g[1]['y_pos']))]*len(each_g[1])
    #     each_g[1]['map_pos'] = counter * len(each_g[1])
        clustered_bar_avg_df = pd.concat([each_g[1], clustered_bar_avg_df])
        counter = counter + 1
    le = LabelEncoder()
    clustered_bar_avg_df = clustered_bar_avg_df.sort_values(['y_pos'])
    clustered_bar_avg_df['sorted_index'] = le.fit_transform(list(clustered_bar_avg_df['avg_pos']))
    # print("clustered_bar_avg_df",clustered_bar_avg_df)
    y_axis_data_df['sorted_index'] = range(0, len(y_axis_data_df))

    data_df = pd.merge(clustered_bar_avg_df, y_axis_data_df, on="sorted_index")
    data_df ['bar_count']= data_df.groupby('cluster').cumcount().add(1)
    data_df['bar_count'] = 'bar' + data_df['bar_count'].astype(str)

    # print("data_df*********",data_df)
    data_df =data_df[["char","bar_count","values"]]
    data_df.rename(columns = {'char':'Y_axis_values','values':'Bar_values'}, inplace = True)
    data_df.drop_duplicates(keep='first',inplace=True)  
    # print(data_df)

    return data_df


def data_extraction_grouped_bar_approach2(image_croping_res):
    x_axis_data_df = x_axis_value_extraction(image_croping_res)
    y_axis_data_df = y_axis_value_extraction(image_croping_res)
    bar_coordinates = bar_coordinates_detection(image_croping_res)
    bar_cal_values = bar_value_calculation(x_axis_data_df, bar_coordinates)
    bar_values = mappig_bar_y_and_bar_values_2(bar_coordinates,bar_cal_values)
    final_df = mapping_yaxis_barvalues_2(y_axis_data_df,bar_values)
    print("*********************************************************************")

    # print(final_df)
    return final_df


#*****main function *****************************************************************************************

def data_extr_hori_grouped(image_croping_res):
    print("******************************************************************************************")   
    print("[INFO] Data extraction in progress...")
    # plt.imshow(image_croping_res['y_axis'][0])
    # plt.show()
    # plt.imshow(image_croping_res['x_axis'][0])
    # plt.show()
    # plt.imshow(image_croping_res['all_bars'][0])
    # plt.show() )

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
        print("[INFO] GROUPED BAR GRAPH WITH VALUES")
        final_df = data_extraction_grouped_bar_approach1(image_croping_res)
    else:
        print("[INFO] GROUPED BAR GRAPH WITH NO VALUES")
        final_df = data_extraction_grouped_bar_approach2(image_croping_res)

    return final_df



# #*****main function for testing*****************************************************************************************

# if __name__ == "__main__":

#     import time
#     from graph_feature_detection import graph_feature_detection_pipeline

#     path = "/home/ideapoke/Desktop/Work/yolo_implementation/Horizonta_graph/grouped_horizontal/Chart_classification_model_horizontal/images/WITH_values.png"
#     # path = "/home/ideapoke/Desktop/Work/yolo_implementation/Horizonta_graph/grouped_horizontal/Chart_classification_model_horizontal/images/WITHOUT_vales2.png"
#     image = cv2.imread(path)
#     image_croping_res = graph_feature_detection_pipeline(image)


#     data_points = char_loc_extraction_val(image_croping_res)
#     print("******Data_points_identified_are:****************\n")
#     print(data_points)
#     print("length of data points:",len(data_points))


    	
#     if len(data_points) > 1:
#     	print("GROUPED BAR GRAPH WITH VALUES")
#     	s_time = time.time()
#     	data_extraction_grouped_bar_approach1(image_croping_res)
#     	print(data_extraction_grouped_bar_approach1(image_croping_res))
#     else:
#     	print("GROUPED BAR GRAPH WITH NO VALUES")
#     	s_time = time.time()
#     	data_extraction_grouped_bar_approach2(image_croping_res)
#     	print(data_extraction_grouped_bar_approach2(image_croping_res))
#     print("Total time consuption:", time.time() - s_time)