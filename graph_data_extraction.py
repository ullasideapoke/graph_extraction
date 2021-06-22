import cv2
import time
from graph_detection import graph_detection_pipeline
from graph_classification import graph_classifation_pipeline
from graph_feature_detection import graph_feature_detection_pipeline
# from data_extraction_process import data_extraction_simple_bar_approach_1
# from data_extraction_process import data_extraction_simple_bar_approach_2
from utils import y_axis_title_extraction_pipeline
from utils import x_axis_title_extraction_pipeline
from utils import graph_title_extraction_pipeline
# from data_extr_verti_simple_bar import data_ext_verti_simple
from data_extr_verti_grouped_bar import data_ext_verti_grouped
# from data_extr_verti_stacked_bar import data_ext_verti_stacked

# from data_extr_hori_simple_bar import data_ext_hori_simple
# from data_extr_hori_grouped_bar import data_extr_hori_grouped
# from data_extr_hori_stacked_bar import data_extr_hori_stacked


# Original color retain
def color_retaintion(graph_feature_details):
    graph_feature_details_cvt = dict()

    for key in graph_feature_details.keys():
        print(key)
        try:
            graph_feature_details_cvt[key] = [cv2.cvtColor(graph_feature_details[key][0],
                                                           cv2.COLOR_BGR2RGB),
                                              graph_feature_details[key][1]]
        except:
            continue

    return graph_feature_details_cvt


# Vertical Simple bar graph ###############################
def vertical_simple_bar_data_extraction(graph):
    values = dict()
    graph_feature_details_ = graph['graph_feature_details']
    features = graph_feature_details_.keys()
    s_t_time = time.time()

    if 'x_axis' in features or 'all_bars' in features:
        data_points = data_ext_verti_simple(graph_feature_details_)
        # values['x_axis_val'] = x_axis_val
        values['data_points'] = data_points


    if "y_axis_title" in features:
        y_axis_title_image = graph_feature_details_['y_axis_title'][0]
        y_axis_title = y_axis_title_extraction_pipeline(y_axis_title_image)
        values["y_axis_title"] = y_axis_title

    if "x_axis_title" in features:
        x_axis_title_image = graph_feature_details_['x_axis_title'][0]
        x_axis_title = x_axis_title_extraction_pipeline(x_axis_title_image)
        values["x_axis_title"] = x_axis_title

    if "graph_title" in features:
        graph_title_image = graph_feature_details_['graph_title'][0]
        graph_title = graph_title_extraction_pipeline(graph_title_image)
        values["graph_title"] = graph_title

    print("[INFO] Total time taken to extract the data from graph:", time.time() - s_t_time)
    # print("-----------------Result Confirmation: x_axis_val, data_points:", values)

    return values


# Vertical Stacked bar graph ###############################
def vertical_stacked_bar_data_extraction(graph):
    values = dict()
    graph_feature_details_ = graph['graph_feature_details']
    features = graph_feature_details_.keys()
    s_t_time = time.time()

    if 'x_axis' in features or 'all_bars' in features:
        final_df = data_ext_verti_stacked(graph_feature_details_)
        # values['x_axis_val'] = x_axis_val
        values['data_points'] = final_df

    if "y_axis_title" in features:
        y_axis_title_image = graph_feature_details_['y_axis_title'][0]
        y_axis_title = y_axis_title_extraction_pipeline(y_axis_title_image)
        values["y_axis_title"] = y_axis_title

    if "x_axis_title" in features:
        x_axis_title_image = graph_feature_details_['x_axis_title'][0]
        x_axis_title = x_axis_title_extraction_pipeline(x_axis_title_image)
        values["x_axis_title"] = x_axis_title

    if "graph_title" in features:
        graph_title_image = graph_feature_details_['graph_title'][0]
        graph_title = graph_title_extraction_pipeline(graph_title_image)
        values["graph_title"] = graph_title

    if "bar_class" in features:
        pass

    print("[INFO] Total time taken to extract the data from graph:", time.time() - s_t_time)
    # print("-----------------Result Confirmation: x_axis_val, data_points:", values)

    return values

# Vertical Grouped bar graph ###############################
def vertical_grouped_bar_data_extraction(graph):
    values = dict()
    graph_feature_details_ = graph['graph_feature_details']
    features = graph_feature_details_.keys()
    s_t_time = time.time()

    if 'x_axis' in features or 'all_bars' in features:
        final_df = data_ext_verti_grouped(graph_feature_details_)
        # values['x_axis_val'] = x_axis_val
        values['data_points'] = final_df

    if "y_axis_title" in features:
        y_axis_title_image = graph_feature_details_['y_axis_title'][0]
        y_axis_title = y_axis_title_extraction_pipeline(y_axis_title_image)
        values["y_axis_title"] = y_axis_title

    if "x_axis_title" in features:
        x_axis_title_image = graph_feature_details_['x_axis_title'][0]
        x_axis_title = x_axis_title_extraction_pipeline(x_axis_title_image)
        values["x_axis_title"] = x_axis_title

    if "graph_title" in features:
        graph_title_image = graph_feature_details_['graph_title'][0]
        graph_title = graph_title_extraction_pipeline(graph_title_image)
        values["graph_title"] = graph_title

    if "bar_class" in features:
        pass

    print("[INFO] Total time taken to extract the data from graph:", time.time() - s_t_time)
    # print("-----------------Result Confirmation: x_axis_val, data_points:", values)

    return values


# Horizontal Simple bar graph ###############################
def horizontal_simple_bar_data_extraction(graph):
    values = dict()
    graph_feature_details_ = graph['graph_feature_details']
    features = graph_feature_details_.keys()
    s_t_time = time.time()

    # With values ******************
    if 'y_axis' in features or 'all_bars' in features:
        # pass
        final_df = data_ext_hori_simple(graph_feature_details_)
        # values['y_axis_val'] = y_axis_val
        values['data_points'] = final_df

        # With no values ******************
        # if len(data_points) == 0:
        #     pass  # -------------------------- data_ext_hori_simple -------------------------- #
            # values['y_axis_val'] = y_axis_val
            # values['data_points'] = data_points

    if "y_axis_title" in features:
        y_axis_title_image = graph_feature_details_['y_axis_title'][0]
        y_axis_title = y_axis_title_extraction_pipeline(y_axis_title_image)
        values["y_axis_title"] = y_axis_title

    if "x_axis_title" in features:
        x_axis_title_image = graph_feature_details_['x_axis_title'][0]
        x_axis_title = x_axis_title_extraction_pipeline(x_axis_title_image)
        values["x_axis_title"] = x_axis_title

    if "graph_title" in features:
        graph_title_image = graph_feature_details_['graph_title'][0]
        graph_title = graph_title_extraction_pipeline(graph_title_image)
        values["graph_title"] = graph_title

    print("[INFO] Total time taken to extract the data from graph:", time.time() - s_t_time)
    # print("-----------------Result Confirmation: x_axis_val, data_points:", values)

    return values


# Horizontal Stacked bar graph ###############################
def horizontal_stacked_bar_data_extraction(graph):
    values = dict()
    print(graph.keys())
    graph_feature_details_ = graph['graph_feature_details']
    features = graph_feature_details_.keys()
    s_t_time = time.time()

    # With values ******************
    if 'y_axis' in features or 'all_bars' in features:
        # pass
        final_df = data_extr_hori_stacked(graph_feature_details_)

        # values['y_axis_val'] = y_axis_val
        values['data_points'] = final_df

        # With no values ******************
        # if len(data_points) == 0:
        #     pass
            # values['y_axis_val'] = y_axis_val
            # values['data_points'] = data_points

    if "y_axis_title" in features:
        y_axis_title_image = graph_feature_details_['y_axis_title'][0]
        y_axis_title = y_axis_title_extraction_pipeline(y_axis_title_image)
        values["y_axis_title"] = y_axis_title

    if "x_axis_title" in features:
        x_axis_title_image = graph_feature_details_['x_axis_title'][0]
        x_axis_title = x_axis_title_extraction_pipeline(x_axis_title_image)
        values["x_axis_title"] = x_axis_title

    if "graph_title" in features:
        graph_title_image = graph_feature_details_['graph_title'][0]
        graph_title = graph_title_extraction_pipeline(graph_title_image)
        values["graph_title"] = graph_title

    if "bar_class" in features:
        pass

    print("[INFO] Total time taken to extract the data from graph:", time.time() - s_t_time)
    # print("-----------------Result Confirmation: x_axis_val, data_points:", values)

    return values


# Horizontal Grouped bar graph ###############################
def horizontal_grouped_bar_data_extraction(graph):
    values = dict()
    graph_feature_details_ = graph['graph_feature_details']
    features = graph_feature_details_.keys()
    s_t_time = time.time()

    # With values ******************
    if 'y_axis' in features or 'all_bars' in features:
        # pass # -------------- data_extr_hori_grouped -------------- #
        final_df = data_extr_hori_grouped(graph_feature_details_)
        # values['y_axis_val'] = y_axis_val
        values['data_points'] = final_df

        # With no values ******************
        # if len(data_points) == 0:
        #     pass
            # values['y_axis_val'] = y_axis_val
            # values['data_points'] = data_points

    if "y_axis_title" in features:
        y_axis_title_image = graph_feature_details_['y_axis_title'][0]
        y_axis_title = y_axis_title_extraction_pipeline(y_axis_title_image)
        values["y_axis_title"] = y_axis_title

    if "x_axis_title" in features:
        x_axis_title_image = graph_feature_details_['x_axis_title'][0]
        x_axis_title = x_axis_title_extraction_pipeline(x_axis_title_image)
        values["x_axis_title"] = x_axis_title

    if "graph_title" in features:
        graph_title_image = graph_feature_details_['graph_title'][0]
        graph_title = graph_title_extraction_pipeline(graph_title_image)
        values["graph_title"] = graph_title

    if "bar_class" in features:
        pass

    print("[INFO] Total time taken to extract the data from graph:", time.time() - s_t_time)
    # print("-----------------Result Confirmation: x_axis_val, data_points:", values)

    return values


# main function
def graph_dtc_cls_ft_dtc_de(img):
    s_time = time.time()
    all_graph_details = dict()

    # GRAPH DETECTION FROM PAGES ------------------------------------------------------------- #
    s_t_time = time.time()
    graph_image = graph_detection_pipeline(img)

    for graph_count in range(len(graph_image)):
        graph_details = dict()

        new_image = list(graph_image.values())[graph_count]
        print("[INFO] Total time taken to detect the graph", time.time() - s_t_time)
        print("[INFO] -----------------Result Confirmation [graph_image]:", graph_image.keys())
        graph_type = list(graph_image.keys())[graph_count]
        graph_details['graph'] = graph_image[graph_type]
        graph_details['image_type'] = graph_type

        # ***************************************************************************** #
        # Only Vertical Bar Graph image will be sent ---------------------------------- #
        # ***************************************************************************** #

        # GRAPH IMAGE CLASSIFICATION ----------------------------------------------------------------------------- #
        if graph_details['image_type'][:-2] == "Vertical Bar Graph":
            s_t_time = time.time()

            graph_class_details = graph_classifation_pipeline(new_image, "Vertical Bar Graph")
            print("[INFO] Total time taken to classify the image", time.time() - s_t_time)
            print("[INFO] -----------------Result Confirmation [graph_details]:", graph_class_details.keys())
            graph_sub_type = list(graph_class_details.keys())
            graph_details['graph_sub_type'] = graph_sub_type

            # ************************************************ #
            # Simple bar graph ############################### #
            # ************************************************ #

            # GRAPH FEATURE DETECTION ------------------------------------------------------------- #        
            if graph_details['graph_sub_type'][0] == 'simple_bar':
                s_t_time = time.time()
                graph_feature_details = graph_feature_detection_pipeline(new_image, "Vertical Bar Graph", "simple_bar")
                graph_feature_details = color_retaintion(graph_feature_details)
                print("[INFO] Total time taken to detect the features of graph:", time.time() - s_t_time)
                print("[INFO] -----------------Result Confirmation [graph_feature_details]:", graph_feature_details.keys())
                graph_details['graph_feature_details'] = graph_feature_details
                print("\ngraph_feature_details.keys()", graph_feature_details.keys(), "\n")

                # DATA EXTRACTION FROM GRAPH IMAGE ------------------------------------------------------------- #
                extracted_data = vertical_simple_bar_data_extraction(graph_details)
                # print(extracted_data)
                graph_details["extracted_data"] = extracted_data

            # ************************************************ #
            # Stacked bar graph ############################## #
            # ************************************************ #

            # GRAPH FEATURE DETECTION ------------------------------------------------------------- #
            if graph_details['graph_sub_type'][0] == 'stacked_bar':
                s_t_time = time.time()
                graph_feature_details = graph_feature_detection_pipeline(new_image, "Vertical Bar Graph", "stacked_bar")
                graph_feature_details = color_retaintion(graph_feature_details)
                print("[INFO] Total time taken to detect the features of graph:", time.time() - s_t_time)
                print("[INFO] -----------------Result Confirmation [graph_feature_details]:", graph_feature_details.keys())
                graph_details['graph_feature_details'] = graph_feature_details
                print("\ngraph_feature_details.keys()", graph_feature_details.keys(), "\n")

                # DATA EXTRACTION FROM GRAPH IMAGE ------------------------------------------------------------- #
                extracted_data = vertical_stacked_bar_data_extraction(graph_details)
                print(extracted_data)
                graph_details["extracted_data"] = extracted_data

            # ************************************************ #
            # Grouped bar graph ###############################
            # ************************************************ #

            # GRAPH FEATURE DETECTION ------------------------------------------------------------- #        
            if graph_details['graph_sub_type'][0] == 'grouped_bar':
                s_t_time = time.time()
                graph_feature_details = graph_feature_detection_pipeline(new_image, "Vertical Bar Graph", "grouped_bar")
                graph_feature_details = color_retaintion(graph_feature_details)
                print("[INFO] Total time taken to detect the features of graph:", time.time() - s_t_time)
                print("[INFO] -----------------Result Confirmation [graph_feature_details]:", graph_feature_details.keys())
                graph_details['graph_feature_details'] = graph_feature_details
                print("\ngraph_feature_details.keys()", graph_feature_details.keys(), "\n")

                # DATA EXTRACTION FROM GRAPH IMAGE ------------------------------------------------------------- #
                extracted_data = vertical_grouped_bar_data_extraction(graph_details)
                print(extracted_data)
                graph_details["extracted_data"] = extracted_data

        # ***************************************************************************** #
        # Only Horizontal Bar Graph image will be sent -------------------------------- #
        # ***************************************************************************** #

        # GRAPH IMAGE CLASSIFICATION ------------------------------------------------------------------------------- #
        if graph_details['image_type'][:-2] == "Horizontal Bar Graph":
            s_t_time = time.time()
            graph_class_details = graph_classifation_pipeline(new_image, "Horizontal Bar Graph")
            print("[INFO] Total time taken to classify the image", time.time() - s_t_time)
            print("[INFO] -----------------Result Confirmation [graph_details]:", graph_class_details.keys())
            graph_sub_type = list(graph_class_details.keys())
            graph_details['graph_sub_type'] = graph_sub_type

            # ************************************************ #
            # Simple bar graph ############################### #
            # ************************************************ #

            # GRAPH FEATURE DETECTION ------------------------------------------------------------- #
            if graph_details['graph_sub_type'][0] == 'simple_bar':
                s_t_time = time.time()
                graph_feature_details = graph_feature_detection_pipeline(new_image, "Horizontal Bar Graph", "simple_bar")
                graph_feature_details = color_retaintion(graph_feature_details)
                print("[INFO] Total time taken to detect the features of graph:", time.time() - s_t_time)
                print("[INFO] -----------------Result Confirmation [graph_feature_details]:", graph_feature_details.keys())
                graph_details['graph_feature_details'] = graph_feature_details
                print("\ngraph_feature_details.keys()", graph_feature_details.keys(), "\n")

                # DATA EXTRACTION FROM GRAPH IMAGE ------------------------------------------------------------- #
                extracted_data = horizontal_simple_bar_data_extraction(graph_details)
                print(extracted_data)
                graph_details["extracted_data"] = extracted_data

            # ************************************************ #
            # Stacked bar graph ###############################
            # ************************************************ #

            # GRAPH FEATURE DETECTION ------------------------------------------------------------- #
            if graph_details['graph_sub_type'][0] == 'stacked_bar':
                s_t_time = time.time()
                graph_feature_details = graph_feature_detection_pipeline(new_image, "Horizontal Bar Graph", "stacked_bar")
                graph_feature_details = color_retaintion(graph_feature_details)
                print("[INFO] Total time taken to detect the features of graph:", time.time() - s_t_time)
                print("[INFO] -----------------Result Confirmation [graph_feature_details]:", graph_feature_details.keys())
                graph_details['graph_feature_details'] = graph_feature_details
                print("\ngraph_feature_details.keys()", graph_feature_details.keys(), "\n")

                # DATA EXTRACTION FROM GRAPH IMAGE ------------------------------------------------------------- #
                extracted_data = horizontal_stacked_bar_data_extraction(graph_details)
                print(extracted_data)
                graph_details["extracted_data"] = extracted_data

            # ************************************************ #
            # Grouped bar graph ############################## #
            # ************************************************ #

            # GRAPH FEATURE DETECTION ------------------------------------------------------------- #
            if graph_details['graph_sub_type'][0] == 'grouped_bar':
                s_t_time = time.time()
                graph_feature_details = graph_feature_detection_pipeline(new_image, "Horizontal Bar Graph", "grouped_bar")
                graph_feature_details = color_retaintion(graph_feature_details)
                print("[INFO] Total time taken to detect the features of graph:", time.time() - s_t_time)
                print("[INFO] -----------------Result Confirmation [graph_feature_details]:", graph_feature_details.keys())
                graph_details['graph_feature_details'] = graph_feature_details
                print("\ngraph_feature_details.keys()", graph_feature_details.keys(), "\n")

                # DATA EXTRACTION FROM GRAPH IMAGE ------------------------------------------------------------- #
                extracted_data = horizontal_grouped_bar_data_extraction(graph_details)
                print(extracted_data)
                graph_details["extracted_data"] = extracted_data
    
        all_graph_details[graph_type] = graph_details

    print()
    print("****************************************************************")
    print("[INFO] Total time taken for whole process", time.time() - s_time)
    print("****************************************************************")

    return all_graph_details


if __name__ == "__main__":
    input_file_path = "/home/pushpendu/Documents/All_task/market_related_information/image_data_extraction/yolo_python_loading/flow_test/v2/data_extration_complete_flow/test/test_15.jpg"
    img_ = cv2.imread(input_file_path)
    graph_details_ = graph_dtc_cls_ft_dtc_de(img_)
    print("\n\nExtracted Data")
    data = graph_details_['extracted_data']
    print(data)
