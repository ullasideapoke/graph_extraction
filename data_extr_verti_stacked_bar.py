from __future__ import print_function

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

import cv2
from scipy import stats
from scipy import stats
import boto3
# import re

import PIL
from PIL import Image
import re
import math
from collections import Counter
import sys 

#Libraries to extract named colors from RGB value
from scipy.spatial import KDTree
import webcolors
from webcolors import CSS3_HEX_TO_NAMES, hex_to_rgb

from statistics import mean
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

import ast
from statistics import mean

import binascii
import struct
import scipy
import scipy.misc
import scipy.cluster


import time
import warnings
warnings.filterwarnings("ignore")

# AWS OCR
print("[INFO] AWS Rekognition OCR Model loaded")   
client = boto3.client('rekognition')

# def closest(list2, K):
#     return list2[min(range(len(list2)), key = lambda i: abs(list2[i]-K))]  
# def Sort_Tuple(tup): 
#     return(sorted(tup, key = lambda x: x[0]))  
# def listOfTuples(l1, l2):
#     return list(map(lambda x, y:(x,y), l1, l2))

def find_number(text):
    num = re.findall(r'[0-9]+',text)
    return " ".join(num)

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

            if top < 0 or right < 0 or bottom <0:
                pass
            
            else:
                final_text.append((text['DetectedText'],(int(left),int(top),int(right),int(bottom))))

    if len(final_text) == 0:
        final_text.append(("no_text",(0,0,0,0)))

    return final_text


#  1. Stacked bar with values #################################################################################

def x_axis_value_extraction(image_croping_res):
    # rotated_image = image_rotation(image_croping_res['x_axis'][0])
    x_axis_text = aws_rekognition_ocr(image_croping_res['x_axis'][0])
    # print("x_axis_text",x_axis_text)   
    x_axis_text1 = text_allignment_horizontal(x_axis_text)
    x_axis_text2=[]
    for i in range(len(x_axis_text1)):
        x_axis_text2.append([x_axis_text1[i][0],x_axis_text1[i][1]])
    x_axis_data_df = pd.DataFrame(x_axis_text2, columns=['char', 'coordinates'])
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
        
#Function to identify the values on all_bars/bar values***************************************************************
        
def bar_value_extraction(image_croping_res,x_axis_data_df):
    bar_value_text = aws_rekognition_ocr(image_croping_res['all_bars'][0])

    if bar_value_text == [] or len(bar_value_text)<=2:
        bar_value_df =[]
    else:

        # print("bar_value_text",bar_value_text)
        bar_value_df = pd.DataFrame(bar_value_text, columns=['char', 'coordinates'])
        bar_value_df['coordinates'].tolist()
        pd.DataFrame(bar_value_df['coordinates'].tolist(), index=bar_value_df.index)
        # print("bar_value_df",bar_value_df)
        bar_value_df[['left', 'top', 'right', 'bottom']] = pd.DataFrame(bar_value_df['coordinates'].tolist(), index=bar_value_df.index)
        bar_value_df=bar_value_df[['char', 'left', 'top', 'right', 'bottom']]
        if (bar_value_df['char'][0].find('%') != -1):
            print ("All values in percentage(%) ")
        else:
            print ("All values in Integer")
        bar_value_df = bar_value_df.sort_values('left',ignore_index=True)
        # bar_value_df['char']=bar_value_df['char'].apply(lambda x: find_number(x))
        # print("bar_value_df",bar_value_df)
        bar_value_keys = bar_value_df['left']
        bar_value_values = bar_value_df['char']

        cluster_length = len(x_axis_data_df)
        # print("cluster_length",cluster_length)
        model_data = [[i,1] for i in list(bar_value_keys)]

        kmeans = KMeans(n_clusters=cluster_length, random_state=0).fit(model_data)
        bar_stack = [[i,j,k] for i,j,k in zip(list(bar_value_keys), list(bar_value_values), list(kmeans.labels_))]
        cluster_data=[]
        for idx in range(len(bar_stack)):
            cluster_data.append(bar_stack[idx][2])
        bar_value_df["cluster"]=cluster_data  
        # print("before",bar_value_df)

        bar_value_df.sort_values(['cluster', 'top'], ascending=[True, True],inplace=True)

        bar_value_df = checking_for_total_values(bar_value_df,image_croping_res)
        # print("after",bar_value_df)


    return bar_value_df

def checking_for_total_values(bar_value_df,image_croping_res):
    length_text1 = aws_rekognition_ocr(image_croping_res['legend'][0])
    legend_text1 = text_allignment_verticle(length_text1)
    length_legend = len(legend_text1)
    # print("length_legend",length_legend)
    dups= bar_value_df.pivot_table(index = ['cluster'], aggfunc = 'size' )
    avg_cluster_length = round(dups.mean( skipna = True))
    # print("avg_cluster_length",avg_cluster_length)

    if avg_cluster_length>length_legend:
        firsts = bar_value_df.groupby(bar_value_df.cluster).first()
        firsts["cluster"] = firsts.index
    #     print(firsts)
        cond = bar_value_df['char'].isin(firsts['char'])
    #     print(cond)
        bar_value_df.drop(bar_value_df[cond].index, inplace = True)  
        # print(bar_value_df)
        return bar_value_df
    else:
        # print(bar_value_df)
        return bar_value_df

def mapping_bar_coordinates_and_bar_values1(bar_value_df):
    bar_value_values = list(bar_value_df["char"])
    bar_value_keys=list(bar_value_df['left'])
    bar_value_top=list(bar_value_df['top'])
#     print(bar_value_keys,bar_value_values,bar_value_top)
    
    return bar_value_keys,bar_value_values,bar_value_top

# Mapping bar values and yaxis data into single data frame
def mapping_legend_and_data_1(x_axis_data_df,bar_value_keys,bar_value_values,bar_value_top,legend_df):

    model_data = [[i,1] for i in list(bar_value_keys)] #bar xaxis val & i,1 if for 2d araay input for kmeans
    if len(bar_value_keys) == len(x_axis_data_df):
        cluster_length = len(x_axis_data_df)
    elif len(bar_value_keys) < len(x_axis_data_df):
        cluster_length = len(bar_value_keys)
    else:
        cluster_length = len(x_axis_data_df)
    # print("cluster_length",cluster_length)       


    kmeans = KMeans(n_clusters=cluster_length, random_state=0).fit(model_data)
    clustered_bar = [[i, j, k,l] for i, j, k,l in zip(list(bar_value_keys), list(bar_value_values), list(kmeans.labels_),list(bar_value_top))]



    clustered_bar_df = pd.DataFrame(clustered_bar,columns=["char","values","cluster","top"])
    clustered_bar_df.sort_values(['cluster', 'top'], ascending=[True, True], inplace=True)

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
    clustered_bar_avg_df.sort_values(['sorted_index', 'top'], ascending=[True, True],inplace=True)

    # print("clustered_bar_avg_df",clustered_bar_avg_df)

    x_axis_data_df['sorted_index'] = range(0, len(x_axis_data_df))
    data_df = pd.merge(clustered_bar_avg_df, x_axis_data_df, on="sorted_index")
    data_df ['stack_count']= data_df.groupby('cluster').cumcount().add(1)
    data_df['stack_count'] = 'stack' + data_df['stack_count'].astype(str)
    # print("data_df*********",data_df)
    data_df =data_df[["char_y","stack_count","values"]]
    data_df.drop_duplicates(keep='first',inplace=True)  
    stack_list=[]
    for i in range(len(legend_df)):
        stack_list.append("stack" + str(i+1))
#     print(stack_list)
    legend_df["stack_count"]=stack_list
    # print(legend_df)
    data_df = pd.merge(data_df,legend_df,on ='stack_count',how ='left')
    data_df.rename(columns = {'char_y':'char_x'}, inplace = True)
    data_df = data_df[["char_x","char","L_color","values"]]

#     print(data_df)

    return data_df


# Pipeline for stacked bar with values
def data_extraction_stacked_bar_approach1(image_croping_res):
    x_axis_data_df = x_axis_value_extraction(image_croping_res)
    bar_value_df = bar_value_extraction(image_croping_res,x_axis_data_df)
    bar_value_keys = mapping_bar_coordinates_and_bar_values1(bar_value_df)[0]
    bar_value_values = mapping_bar_coordinates_and_bar_values1(bar_value_df)[1]
    bar_value_top = mapping_bar_coordinates_and_bar_values1(bar_value_df)[2]
    background= background_rgb(image_croping_res)
    type_legend = identify_legend(image_croping_res)

    legend_colors_df = legend_color(image_croping_res,background,type_legend)
    legend_text_df = legend_text(image_croping_res,type_legend)
    legend_df=mapping_legend_color_text(legend_text_df,legend_colors_df,type_legend)
    final_df = mapping_legend_and_data_1(x_axis_data_df,bar_value_keys,bar_value_values,bar_value_top,legend_df)   
    # print("*********************stack with values******************************")
#     print(final_df)
    return final_df


# 2.STACKED BAR GRAPH WITH NO VALUES#############################################################################################--->

def y_axis_value_extraction(image_croping_res):
    # rotated_image = image_rotation(image_croping_res['y_axis'][0])
#     plt.imshow(image_croping_res['y_axis'][0])
#     plt.show()
    y_axis_text = aws_rekognition_ocr(image_croping_res['y_axis'][0])
    # print("1st",y_axis_text)
    y_axis_text = text_allignment_verticle(y_axis_text)
    # print("2nd",y_axis_text)
    y_axis_data_df = pd.DataFrame(y_axis_text, columns=['char', 'coordinates'])
#     print("data_type",y_axis_data_df.dtypes)
    y_axis_data_df['coordinates'].tolist()
    pd.DataFrame(y_axis_data_df['coordinates'].tolist(), index=y_axis_data_df.index)
    y_axis_data_df[['left', 'top', 'right', 'bottom']] = pd.DataFrame(y_axis_data_df['coordinates'].tolist(), index=y_axis_data_df.index)
    y_axis_data_df=y_axis_data_df[['char', 'left', 'top', 'right', 'bottom']]
    y_axis_data_df = y_axis_data_df.sort_values('top',ignore_index=True)
#     if (y_axis_data_df['char'][0].find('%') != -1):
#         print ("All values in percentage(%) ")
#     else:
#         print ("All values in Integer")
    # print(y_axis_data_df)
    y_axis_data_df['char']  = y_axis_data_df['char'].astype(str)
    y_axis_data_df['char']= y_axis_data_df['char'].str.replace('o','0')  
    y_axis_data_df['char']= y_axis_data_df['char'].str.replace('O','0')
    y_axis_data_df['char']= y_axis_data_df['char'].str.replace(' ','')
#     y_axis_data_df['char']= y_axis_data_df['char'].str.replace('K','000')  
#     y_axis_data_df['char']= y_axis_data_df['char'].str.replace('k','000')  
#     y_axis_data_df['char']= y_axis_data_df['char'].str.replace('M','000000')  
#     y_axis_data_df['char']= y_axis_data_df['char'].str.replace('m','000000')  
#     y_axis_data_df['char']= y_axis_data_df['char'].str.replace('$','')
    y_axis_data_df['char']=y_axis_data_df['char'].apply(lambda x: find_number(x))

    # print(y_axis_data_df)
    y_axis_data_df['char'] = y_axis_data_df['char'].astype(int)
    
    # print("*************y_axis_data************************")   
    # print(y_axis_data_df)
    

    return y_axis_data_df


#Stage 1 : extract color names 

# extract the RGB values from stack
def all_stacks(image_croping_res): 
    stack_array = []
    stack = []

    for key in image_croping_res.keys():
        if "stack" in key:
            stack.append(key)
            stack_array.append(image_croping_res[key][0])
            # stack_array.append(cv2.cvtColor(image_croping_res[key][0],cv2.COLOR_BGR2RGB))
    return stack_array

def get_RGB_value(stack_array): 


    rgb_value=[]
    for j in range(len(stack_array)):
        NUM_CLUSTERS = 5
        im = stack_array[j]
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

def closest_colour(requested_colour):
    min_colours = {}
    for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_colour[0]) ** 2
        gd = (g_c - requested_colour[1]) ** 2
        bd = (b_c - requested_colour[2]) ** 2
        min_colours[(rd + gd + bd)] = name
    return min_colours[min(min_colours.keys())]

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
        named_color.append(["stack"+str(i+1), get_colour_name(rgb_values[i])])
    named_color = pd.DataFrame(named_color,columns=["stack_num","stack_color"])
#     print(named_color)
    return named_color    


#Stage 2 : extract values using stack and y_axis

def val_per_pixels(y_axis_data_df):
    total_pixel = max(y_axis_data_df["top"])-min(y_axis_data_df["top"])
    total_value =  max(y_axis_data_df["char"].astype(int))-min(y_axis_data_df["char"].astype(int))
    val_per_pixel = total_value/ total_pixel
    # print("val_per_pixel",val_per_pixel)
       
    return val_per_pixel 

def mapping_bar_coordinates_and_bar_values2(image_croping_res,val_per_pixel):

    stack_array = []
    stack = []
    for key in image_croping_res.keys():
        if "stack" in key:
            stack.append(key)
            stack_array.append(image_croping_res[key][1])

    stack_array1=[]
    for i in range(len(stack_array)):
        stack_array1.append(["stack"+str(i+1),stack_array[i],round(stack_array[i][3]*val_per_pixel)])
#     print(stack_array1)
    stack_value_keys = []
    stack_value_values=[]
    stack_value_y=[]
    stack_num=[]
    for j in range (len(stack_array1)):
        stack_value_keys.append(stack_array1[j][1][0])
        stack_value_values.append(stack_array1[j][2])
        stack_value_y.append(stack_array1[j][1][1])
        stack_num.append(stack_array1[j][0])
    # print("stack_value_keys",stack_value_keys)
    # print("stack_value_values",stack_value_values)
    # print("stack_value_y",stack_value_y)
    # print("stack_num",stack_num)


    return stack_value_keys,stack_value_values,stack_value_y,stack_num


def mapping_all_values(x_axis_data_df,stack_value_keys,stack_value_values,stack_value_y,stack_num,named_color):
    cluster_length = len(x_axis_data_df)
    model_data = [[i,1] for i in list(stack_value_keys)]
  

    kmeans = KMeans(n_clusters=cluster_length, random_state=0).fit(model_data)
    clustered_stack = [[i,j,k,l,m] for i,j,k,l,m in zip(list(stack_value_keys), list(stack_value_values), list(kmeans.labels_),list(stack_value_y),list(stack_num))]
    # print(clustered_stack)
    clustered_stack_df = pd.DataFrame(clustered_stack,columns=["char","values","cluster","top","stack_num"])
    clustered_stack_df.sort_values(["cluster", "top"], ascending=[True, True], inplace=True)
    counter = 1
    clustered_stack_avg_df = pd.DataFrame()
    for each_g in clustered_stack_df.groupby(['cluster']):
        each_g[1]['avg_pos'] = [Average(list(each_g[1]['char']))]*len(each_g[1])
        clustered_stack_avg_df = pd.concat([each_g[1], clustered_stack_avg_df])
        counter = counter + 1
    le = LabelEncoder()
    clustered_stack_avg_df = clustered_stack_avg_df.sort_values(['char'])
    clustered_stack_avg_df['sorted_index'] = le.fit_transform(list(clustered_stack_avg_df['avg_pos']))
    x_axis_data_df['sorted_index'] = range(0, len(x_axis_data_df))
    data_df = pd.merge(clustered_stack_avg_df, x_axis_data_df, on="sorted_index")
    data_df =data_df[["char_y","values","stack_num"]]
    data_df = pd.merge(data_df, named_color, on="stack_num")
    data_df.rename(columns = {'char_y':'char_x'}, inplace = True)
    data_df.drop(['stack_num'], axis = 1, inplace = True)
    data_df = data_df[["char_x","stack_color","values"]]
    # print("data_df",data_df)
    return data_df

#Stage 3 : codes for color identification, legend color and legend text detection

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

def identify_legend(image_croping_res):

    if 'legend' in image_croping_res:

        legend_texts = aws_rekognition_ocr(image_croping_res['legend'][0])
        for i in range(len(legend_texts)-1):
            if legend_texts[i+1][1][1] - legend_texts[i][1][1] in range(-5,5):
                type_legend1 = "horizontal"
            else:
                type_legend1 = "verticle"
        if "verticle" in type_legend1:
            type_legend =  "verticle"
            # legend_texts = text_allignment_verticle(legend_texts)
        else:
            type_legend =  "horizontal"
            # legend_texts = text_allignment_horizontal(legend_texts)

        print("legend type is :" ,type_legend)

    else:
        type_legend =  "verticle"

    return type_legend

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
#     print("legend_color_values",legend_color_values)
    legend_colors_df = named_colors(legend_color_values)
    legend_colors_df["coordinates"] = legend_color_loc
    legend_colors_df['coordinates'].tolist()
    pd.DataFrame(legend_colors_df['coordinates'].tolist(), index=legend_colors_df.index)
    legend_colors_df[['left', 'top', 'right', 'bottom']] = pd.DataFrame(legend_colors_df['coordinates'].tolist(), index=legend_colors_df.index)
    
    if type_legend == 'verticle':
        legend_colors_df.sort_values('top', ascending=True, inplace=True)
    else:
        legend_colors_df.sort_values('left', ascending=True, inplace=True)  

    print("legend_colors_df",legend_colors_df)
    
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
    # print(legend_color_loc)
    legend_text_val=[]
    for i in range(len(legend_text_array)):
        legend_text_val.append([aws_rekognition_ocr(legend_text_array[i]),legend_color_loc[i]])
    # print("legend_text_val",legend_text_val)
    # legend_text_val = text_allignment_verticle(legend_text_val)
    # print("legend_text_val",legend_text_val)
    # print("text",legend_text_val[2][0][0][0]) 
    # print("coordinates",legend_text_val[0][1])

    column1 =[]
    column2 =[]
    for j in range(len(legend_text_val)):
        column1.append(legend_text_val[j][0][0][0])
        column2.append(legend_text_val[j][1])
    legend_text_df = pd.DataFrame(list(zip(column1, column2)),columns =['text', 'coordinates'])
    legend_text_df['coordinates'].tolist()
    pd.DataFrame(legend_text_df['coordinates'].tolist(), index=legend_text_df.index)
    legend_text_df[['left', 'top', 'right', 'bottom']] = pd.DataFrame(legend_text_df['coordinates'].tolist(), index=legend_text_df.index)
    
    if type_legend == 'verticle':
        legend_text_df.sort_values('top', ascending=True, inplace=True)
    else:
        legend_text_df.sort_values('left', ascending=True, inplace=True)
    print("legend_text_df",legend_text_df)
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
    legend_df =legend_df[["char","stack_color"]]
    legend_df.rename(columns = {'stack_color':'L_color'}, inplace = True)
    # print("legend_df",legend_df)    

    return legend_df

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
    # print("data_df",data_df)
    # print("legend_df",legend_df)

   
    legendcol = list(legend_df["L_color"])
    stackcol= list(data_df["stack_color"])

    
    u_legendcol = unique(legendcol)
    u_stackcol = unique(stackcol)

    n_legendcol = named_color_to_rgb(u_legendcol)
    n_stackcol= named_color_to_rgb(u_stackcol)

    df2 = pd.DataFrame(list(zip(u_stackcol,n_stackcol)),columns =['S_Name','S_RGB'])

    result=[]
    for i in range(len(n_stackcol)):
        test= n_stackcol[i]
        result.append(nearest_color_detection(n_legendcol,test))
#     print(result) 
    color_result = named_colors(result)
    stack_color = color_result["stack_color"].tolist()
    df2["near_rgb"] =result
    df2["stack_color"] =stack_color
    df1 = pd.DataFrame(list(zip(u_legendcol, n_legendcol)),columns =['stack_color', 'L_RGB'])
    color_result = pd.merge(df2,df1,on ='stack_color',how ='left')
    color_result = color_result[['S_Name','stack_color']]
    color_result.rename(columns = {'stack_color':'L_color','S_Name':'stack_color'}, inplace = True)
    # print("color_result",color_result)
    data_df_new = pd.merge(data_df,color_result,on ='stack_color',how ='left')

    final_df = pd.merge(data_df_new,legend_df,on ='L_color',how ='left')
    # print("final_df",final_df)
    final_df = final_df[["char_x","char","stack_color","values"]]
    final_df.rename(columns = {'char_x':'x_axis','char':'category'}, inplace = True)
    final_df.sort_values(['x_axis', 'category'], ascending=[True, True],inplace=True)

    return final_df

def data_extraction_stacked_bar_approach2(image_croping_res):
    x_axis_data_df = x_axis_value_extraction(image_croping_res)
    y_axis_data_df = y_axis_value_extraction(image_croping_res)
    stack_array = all_stacks(image_croping_res)
    stack_rgb_values = get_RGB_value(stack_array)
    val_per_pixel= val_per_pixels(y_axis_data_df)
    stack_named_color = named_colors(stack_rgb_values)
    stack_value_keys=mapping_bar_coordinates_and_bar_values2(image_croping_res,val_per_pixel)[0]
    stack_value_values=mapping_bar_coordinates_and_bar_values2(image_croping_res,val_per_pixel)[1]
    stack_value_y=mapping_bar_coordinates_and_bar_values2(image_croping_res,val_per_pixel)[2]
    stack_num=mapping_bar_coordinates_and_bar_values2(image_croping_res,val_per_pixel)[3]
    background= background_rgb(image_croping_res)
    data_df = mapping_all_values(x_axis_data_df,stack_value_keys,stack_value_values,stack_value_y,stack_num,stack_named_color)
    type_legend = identify_legend(image_croping_res)
    legend_colors_df = legend_color(image_croping_res,background,type_legend)
    legend_text_df = legend_text(image_croping_res,type_legend)
    legend_df=mapping_legend_color_text(legend_text_df,legend_colors_df,type_legend)
    final_df = mapping_legend_and_data_2(legend_df,data_df)
    # print("******************Stack without values**************************")
    # print(final_df)

    return final_df


def data_ext_verti_stacked(image_croping_res):
    print("******************************************************************************************")   
    print("[INFO] Data extraction in progress...")
    # plt.imshow(image_croping_res['y_axis'][0])
    # plt.show()
    # plt.imshow(image_croping_res['x_axis'][0])
    # plt.show()
    # plt.imshow(image_croping_res['all_bars'][0])
    # plt.show() 
    # plt.imshow(image_croping_res['legend'][0])
    # plt.show() 
    # plt.imshow(image_croping_res['legend_color1'][0])
    # plt.show() 
    # plt.imshow(image_croping_res['legend_color2'][0])
    # plt.show() 
    # plt.imshow(image_croping_res['legend_color3'][0])
    # plt.show() 
    # plt.imshow(image_croping_res['legend_color4'][0])
    # plt.show() 
    # plt.imshow(image_croping_res['legend_text1'][0])
    # plt.show() 
    # plt.imshow(image_croping_res['legend_text2'][0])
    # plt.show() 
    # plt.imshow(image_croping_res['legend_text3'][0])
    # plt.show() 
    # plt.imshow(image_croping_res['legend_text4'][0])
    # plt.show() 

    x_axis_data_df = x_axis_value_extraction(image_croping_res)

    data_points = bar_value_extraction(image_croping_res,x_axis_data_df)
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
        print("\nSTACKED BAR GRAPH WITH VALUES")
        final_df = data_extraction_stacked_bar_approach1(image_croping_res)
    else:
        print("\nSTACKED BAR GRAPH WITHOUT VALUES")
        final_df = data_extraction_stacked_bar_approach2(image_croping_res)

    return final_df


# # *****main function for testing*****************************************************************************************
# if __name__ == "__main__":

#     import time
#     from graph_feature_detection import graph_feature_detection_pipeline

#     # path = "//home/ideapoke/Desktop/Work/yolo_implementation/Horizonta_graph/stacked_chart/Chart_classification_model_horizontal/images/1_stacked_full_image.png"
#     path = "/home/ideapoke/Desktop/Work/yolo_implementation/verticle_bar_graph/images/stacked-val-bar-example-3.png"    

#     image = cv2.imread(path)
#     graph_image = "Vertical Bar Graph"
#     graph_sub_type = "stacked_bar"
#     image_croping_res = graph_feature_detection_pipeline(image, graph_image, graph_sub_type)

#     print(data_ext_verti_stacked(image_croping_res))