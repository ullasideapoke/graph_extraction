import re
import time
import math
import numpy as np
import pandas as pd
from scipy import stats
import keras_ocr
import pytesseract

import warnings

warnings.filterwarnings("ignore")

print("[INFO] KERAS OCR Model loading...")
pipeline = keras_ocr.pipeline.Pipeline()


# Average value calculation
def Average(lst):
    try:
        avg = sum(lst) / len(lst)
        return int(avg)
    except:
        return 0


# Noise removal
def remove_punctuation(string):
    regex = re.compile('[^0-9.%]')
    eng_text = regex.sub('', string)
    return eng_text


# Outlier detection
def reject_outliers(data):
    d = 1
    m = np.mean(data)
    s = np.std(data)
    filtered = [e for e in data if (m - d * s < e < m + d * s)]
    return filtered


# SIMPLE BAR GRAPH WITH VALUES    ##################################################################################--->
# KERAS OCR MODEL
# Value reading from x axis
def x_axis_wise_value(image_croping_res_):
    if "x_axis" in image_croping_res_.keys():
        x_axis_val_loc = list()

        images = keras_ocr.tools.read(image_croping_res_['x_axis'][0])
        prediction_groups = pipeline.recognize([images])
        # print("prediction_groups ------>", prediction_groups)
        for i in prediction_groups[0]:
            temp_val = list()
            temp_val.append(i[0])

            for j in i[1][0:2]:
                for k in j:
                    temp_val.append(k)

            x_axis_val_loc.append(temp_val)

        # print("\nx_axis_val_loc---->", x_axis_val_loc)
        x_axis_val_loc_df = pd.DataFrame(x_axis_val_loc, columns=['char', 'left', 'bottom', 'right', 'top'])
        x_axis_val_loc_sorted_df = x_axis_val_loc_df.sort_values('left')

        return x_axis_val_loc_sorted_df


# Pytesseract Model
def char_loc_extraction_val(image_croping_res__):
    val_char = {'1', '2', '3', '4', '5', '6', '7', '8', '9', '0', ".", '%', 'S', "'"}

    config = '-l eng --oem 1 --psm 3'
    # results = pytesseract.image_to_string(image_croping_res['all_bars'], config=config)
    results = pytesseract.image_to_boxes(image_croping_res__['all_bars'][0], config=config)

    final_res = []
    for i in results.split("\n"):
        i_s = i.split(" ")[0:5]

        if len(i_s) > 0:
            if i_s[0] in val_char:
                i_s_m = [i_s[0]]

                for j in i_s[1:]:
                    i_s_m.append(int(j))

                final_res.append(i_s_m)

    char_loc_df = pd.DataFrame(final_res,
                               columns=['char', 'left', 'bottom', 'right', 'top'])

    return char_loc_df


def bottom_point_correction(char_loc_df):
    sorted_point = sorted([int(i) for i in list(char_loc_df['bottom'].unique())])

    corrected_point = dict()
    for idx in range(len(sorted_point) - 1):
        if sorted_point[idx + 1] - sorted_point[idx] == 1:
            corrected_point[sorted_point[idx]] = sorted_point[idx + 1]

    for key in corrected_point.keys():
        char_loc_df['bottom'] = char_loc_df['bottom'].replace([key], corrected_point[key])

    return char_loc_df


def char_grouping(x_axis_val_loc_sorted_df, char_loc_df):
    temp_list = []
    count = 1

    for idx in range(len(x_axis_val_loc_sorted_df)):
        left_loc = math.floor(x_axis_val_loc_sorted_df.loc[idx]['left']) - 20
        right_loc = math.ceil(x_axis_val_loc_sorted_df.loc[idx]['right']) + 20

        for idx_ in range(len(char_loc_df)):
            left_char_loc = int(char_loc_df.loc[idx_]['left'])
            right_char_loc = int(char_loc_df.loc[idx_]['right'])

            if left_loc <= left_char_loc <= right_loc or left_loc < right_char_loc < right_loc:
                char_list = char_loc_df.loc[idx_].tolist()[0:5]
                char_list.append("group" + str(count))
                temp_list.append(char_list)

        count = count + 1

    char_loc_group_df = pd.DataFrame(temp_list, columns=['char', 'left', 'bottom', 'right', 'top', 'group'])

    return char_loc_group_df


def data_points_from_bar_grouped_s(char_loc_group_df):
    data_loc_list = []

    for each_g in char_loc_group_df.groupby(["group"]):
        data = "".join(each_g[1].sort_values("left", ascending=True)['char'].tolist())

        # trying to split the data like 27,.27.12.
        for part_data in data.split(","):
            part_data = part_data.replace("'", ".")
            part_data = part_data.replace("S", "5")

            part_data_num = part_data.replace(".", "")
            part_data_num = part_data_num.replace("%", "")

            if part_data_num.replace(".", "").isdigit():
                left = Average([int(i) for i in each_g[1]['left'].tolist()])
                bottom = Average([int(i) for i in each_g[1]['bottom'].tolist()])
                right = Average([int(i) for i in each_g[1]['right'].tolist()])
                top = Average([int(i) for i in each_g[1]['top'].tolist()])

                part_data = part_data.strip(".")
                part_data = part_data.replace("..", ".")
                data_loc_list.append([part_data, left, bottom, right, top])

    data_loc_df = pd.DataFrame(data_loc_list,
                               columns=['char', 'left', 'bottom', 'right', 'top'])

    data_loc_sorted_df = data_loc_df.sort_values('left')
    data_points_ = data_loc_sorted_df['char'].tolist()

    return data_points_


# ELSE
def data_points_from_bar(char_loc_df):
    data_loc_list = []

    for each_g in char_loc_df.groupby(["bottom"]):
        data = "".join(each_g[1].sort_values("left", ascending=True)['char'].tolist())

        part_data = data.replace("'", ".")
        part_data = part_data.replace("S", "5")

        part_data_num = part_data.replace(".", "")
        part_data_num = part_data_num.replace("%", "")

        if part_data_num.replace(".", "").isdigit():
            left = Average([int(i) for i in each_g[1]['left'].tolist()])
            bottom = Average([int(i) for i in each_g[1]['bottom'].tolist()])
            right = Average([int(i) for i in each_g[1]['right'].tolist()])
            top = Average([int(i) for i in each_g[1]['top'].tolist()])

            part_data = part_data.strip(".")
            data_loc_list.append([part_data, left, bottom, right, top])

    data_loc_df = pd.DataFrame(data_loc_list,
                               columns=['char', 'left', 'bottom', 'right', 'top'])

    data_loc_sorted_df = data_loc_df.sort_values('left')
    data_points__ = data_loc_sorted_df['char'].tolist()

    return data_points__


def data_extraction_simple_bar_approach_1(image_croping_res___):
    # bar_count = len([i for i in image_croping_res___.keys() if 'bar' in i and i != 'all_bars'])
    x_axis_val_loc_sorted_df = x_axis_wise_value(image_croping_res___)
    # print("\nx_axis_val_loc_sorted_df ---", x_axis_val_loc_sorted_df)

    if len(x_axis_val_loc_sorted_df) > 0:
        x_axis_val__ = x_axis_val_loc_sorted_df['char'].tolist()

    char_loc_df = char_loc_extraction_val(image_croping_res___)
    char_loc_df = bottom_point_correction(char_loc_df)
    char_loc_group_df = char_grouping(x_axis_val_loc_sorted_df, char_loc_df)
    data_point = data_points_from_bar_grouped_s(char_loc_group_df)

    if len(data_point) < 2:
        print("[INFO] value not detected approach")
        data_point = data_points_from_bar(char_loc_df)

    return x_axis_val__, data_point


# SIMPLE BAR GRAPH WITH NO VALUES ##################################################################################--->
def char_loc_extraction_y_axis(image_croping_ress):
    val_char = {'1', '2', '3', '4', '5', '6', '7', '8', '9', '0', ".", '%', 'S', "'", "s", "m", "o", "l"}

    config = '-l eng --oem 1 --psm 3'
    results = pytesseract.image_to_boxes(image_croping_ress['y_axis'][0], config=config)

    final_res = []
    for i in results.split("\n"):
        i_s = i.split(" ")[0:5]

        if len(i_s) > 0:
            if i_s[0] in val_char:
                i_s_m = [i_s[0]]

                for j in i_s[1:]:
                    i_s_m.append(int(j))

                final_res.append(i_s_m)

    char_loc_df = pd.DataFrame(final_res,
                               columns=['char', 'left', 'bottom', 'right', 'top'])

    return char_loc_df


def data_points_from_bar_grouped_n_s(char_loc_group_df):
    data_loc_list = []

    for each_g in char_loc_group_df.groupby(["bottom"]):
        data = "".join(each_g[1].sort_values("left", ascending=True)['char'].tolist())

        part_data = data.replace("'", ".")
        part_data = part_data.replace("S", "5")
        part_data = part_data.replace("s", "5")
        part_data = part_data.replace("m", "00")
        part_data = part_data.replace("o", "0")
        part_data = part_data.replace("O", "0")

        part_data_num = part_data.replace(".", "")
        part_data_num = part_data_num.replace("%", "")

        if part_data_num.replace(".", "").isdigit():
            left = Average([int(i) for i in each_g[1]['left'].tolist()])
            bottom = Average([int(i) for i in each_g[1]['bottom'].tolist()])
            right = Average([int(i) for i in each_g[1]['right'].tolist()])
            top = Average([int(i) for i in each_g[1]['top'].tolist()])

            part_data = part_data.strip(".")
            part_data = part_data.replace("..", ".")
            data_loc_list.append([part_data, left, bottom, right, top])

    data_loc_df = pd.DataFrame(data_loc_list,
                               columns=['char', 'left', 'bottom', 'right', 'top'])

    return data_loc_df


def bar_coordinates(image_croping_res_b):
    bar_pos = []
    bar = []
    # print("image_croping_res_b", image_croping_res_b)
    for key in image_croping_res_b.keys():
        if "bar" in key and key != "all_bars":
            bar.append(key)
            bar_pos.append(image_croping_res_b[key][1])

    # print("*** bar_pos ***", bar_pos)
    data = pd.DataFrame(bar_pos, columns=['x', 'y', 'w', 'h'], index=bar)
    data_s = data.sort_values('x')

    return data_s


def bar_value_calculation(y_axis_data_df, data_s):
    dif_h = []
    dif_v = []
    bar_cal_values = []

    for idx in range(len(y_axis_data_df) - 1):
        val_1 = y_axis_data_df.loc[idx]["top"]
        val_2 = y_axis_data_df.loc[idx + 1]["top"]

        dif_h.append(val_2 - val_1)

    dif_h = reject_outliers(dif_h)
    avg_dif = Average(dif_h)

    for idx in range(len(y_axis_data_df) - 1):
        val_1 = float(y_axis_data_df.loc[idx]["char"])
        val_2 = float(y_axis_data_df.loc[idx + 1]["char"])

        dif_v.append(val_2 - val_1)

    dif_val = stats.mode(dif_v)[0][0]

    for i in list(data_s['h']):
        bar_cal_values.append(round((dif_val / avg_dif) * i, 2))

    return bar_cal_values


# Simple graph no value
def data_extraction_simple_bar_approach_2(image_croping_res___):
    x_axis_val_loc_sorted_df = x_axis_wise_value(image_croping_res___)
    x_axis_val_ = x_axis_val_loc_sorted_df['char'].tolist()

    char_loc_df = char_loc_extraction_y_axis(image_croping_res___)
    char_loc_df = bottom_point_correction(char_loc_df)
    # char_loc_group_df = char_grouping(x_axis_val_loc_sorted_df, char_loc_df)
    y_axis_data_df = data_points_from_bar_grouped_n_s(char_loc_df)
    data_s = bar_coordinates(image_croping_res___)
    bar_values = bar_value_calculation(y_axis_data_df, data_s)

    return x_axis_val_, bar_values


# STACKED BAR GRAPH WITH NO VALUES ##################################################################################--->
def bar_value_calculation_stacked(y_axis_data_df, data_s):
    dif_h = []
    dif_v = []
    bar_cal_values = dict()

    for idx in range(len(y_axis_data_df) - 1):
        val_1 = y_axis_data_df.loc[idx]["top"]
        val_2 = y_axis_data_df.loc[idx + 1]["top"]

        dif_h.append(val_2 - val_1)

    dif_h = reject_outliers(dif_h)
    avg_dif = Average(dif_h)

    for idx in range(len(y_axis_data_df) - 1):
        val_1 = float(y_axis_data_df.loc[idx]["char"])
        val_2 = float(y_axis_data_df.loc[idx + 1]["char"])

        dif_v.append(val_2 - val_1)

    dif_val = stats.mode(dif_v)[0][0]

    for idx in data_s.index.tolist():
        height = int(data_s.loc[idx]['h'])
        bar_cal_values[idx] = round((dif_val / avg_dif) * height)

    return bar_cal_values


# Color area calculation
def processLog(imnp):
    bar_colr_div = dict()
    color_count = 1
    h, w = imnp.shape[:2]

    # Get list of unique colours...
    # Arrange all pixels into a tall column of 3 RGB values and find unique rows (colours)
    colours, counts = np.unique(imnp.reshape(-1, 3), axis=0, return_counts=1)

    # Iterate through unique colours
    for index, colour in enumerate(colours):
        count = counts[index]
        proportion = (100 * count) / (h * w)
        if proportion > 1:
            key = "color_" + str(color_count)
            bar_colr_div[key] = [list(colour), count, round(proportion, 2)]
            color_count = color_count + 1

    return bar_colr_div


# Stack value calculation for each bar
def all_bar_value_cal(image_croping_res, y_axis_data_df, data_s):
    bar_colr_div = processLog(image_croping_res["bar2"][0])
    bar_values = bar_value_calculation_stacked(y_axis_data_df, data_s)

    colr_area = sorted([i[2] for i in bar_colr_div.values()], reverse=True)
    all_bar_values = [round(i / 100 * bar_values["bar2"], 2) for i in colr_area]

    return all_bar_values


# Stack value extraction main pipeline
def data_extraction_stacked_bar(image_croping_res___):
    x_axis_val_loc_sorted_df = x_axis_wise_value(image_croping_res___)
    x_axis_val_ = x_axis_val_loc_sorted_df['char'].tolist()

    char_loc_df = char_loc_extraction(image_croping_res___)
    char_loc_df = bottom_point_correction(char_loc_df)
    y_axis_data_df = data_points_from_bar_grouped_n_s(char_loc_df)
    data_s = bar_coordinates(image_croping_res)
    all_bar_values = all_bar_value_cal(image_croping_res, y_axis_data_df, data_s)

    return x_axis_val_, all_bar_values


# Grouped value extraction main pipeline
def data_extraction_grouped_bar(image_croping_res___):
    x_axis_val_loc_sorted_df = x_axis_wise_value(image_croping_res___)
    x_axis_val_ = x_axis_val_loc_sorted_df['char'].tolist()

    char_loc_df = char_loc_extraction(image_croping_res___)
    char_loc_df = bottom_point_correction(char_loc_df)
    y_axis_data_df = data_points_from_bar_grouped_n_s(char_loc_df)
    data_s = bar_coordinates(image_croping_res)
    all_bar_values = all_bar_value_cal(image_croping_res, y_axis_data_df, data_s)

    return x_axis_val_, all_bar_values


if __name__ == "__main__":
    import cv2
    import time
    from graph_feature_detection import graph_feature_detection_pipeline

    path = "/home/pushpendu/Documents/All_task/market_related_information/image_data_extraction/yolo_python_loading/flow_test/v2/data_extration_complete_flow/test/test_15_no_value.jpg"
    image = cv2.imread(path)
    image_croping_res = graph_feature_detection_pipeline(image)
    # print("image_croping_res", image_croping_res.keys())
    graph_sub_type = ['simple_bar']

    # SIMPLE BAR GRAPH WITH VALUES
    data_points = list()

    if graph_sub_type[0] == 'simple_bar':
        s_time = time.time()
        x_axis_val, data_points = data_extraction_simple_bar_approach_1(image_croping_res)

        if len(data_points) == 0:
            # SIMPLE BAR GRAPH WITH NO VALUES
            s_time = time.time()
            x_axis_val, data_points = data_extraction_simple_bar_approach_2(image_croping_res)

        print("Total time consuption:", time.time() - s_time)
        print("x_axis_val, data_points", x_axis_val, data_points)

    # STACKED BAR GRAPH WITH NO VALUES
    if graph_sub_type[0] == 'stacked_bar':
        s_time = time.time()
        data_points = []
        # x_axis_val, data_points = data_extraction_stacked_bar_approach_1(image_croping_res) # dummy line

        if len(data_points) == 0:
            # SIMPLE BAR GRAPH WITH NO VALUES
            # x_axis_val_, data_points = data_extraction_stacked_bar_approach_2(image_croping_res___) # dummy line
            pass

        print("Total time consuption:", time.time() - s_time)
        # print("x_axis_val, all_bar_values", x_axis_val_, all_bar_values)

    # GROUPED BAR GRAPH WITH NO VALUES
    if graph_sub_type[0] == 'group_bar':
        s_time = time.time()
        data_points = []
        # x_axis_val_, data_points = data_extraction_grouped_bar_approach_1(image_croping_res___) # dummy line

        if len(data_points) == 0:
            # SIMPLE BAR GRAPH WITH NO VALUES
            # x_axis_val_, data_points = data_extraction_grouped_bar_approach_2(image_croping_res___) #dummy line
            pass

        print("Total time consuption:", time.time() - s_time)
