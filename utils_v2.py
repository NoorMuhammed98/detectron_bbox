import os
import cv2
import torch,torchvision
import detectron2
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2 import model_zoo
from detectron2.config import get_cfg
import matplotlib.pyplot as plt
import PIL.Image
import numpy as np
from osgeo import gdal
from PIL import Image
import geopandas as gpd
from shapely.geometry import Polygon
import pandas as pd
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

model_pth = r"C:\Users\User-1\Desktop\Data\detectron2\instance_segmentation\output\instance_segmentation\model_final.pth"
threshold = 0.75

def initialize_detectron2(model,thrsh):
    # Set your config file path
    config_file = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config_file))
    # Load the model weights
    cfg.MODEL.WEIGHTS = model
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = thrsh # set a threshold for this model
    cfg.MODEL.DEVICE = "cuda"  # Use CUDA for inference
    # Create a predictor
    predictor = DefaultPredictor(cfg)
    return predictor

def visualizer(image_path):
    im = cv2.imread(image_path)# Perform inference
    predictor = initialize_detectron2(model_pth,threshold)
    outputs = predictor(im)

    # Convert image from BGR to RGB (if needed)
    im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    # Get instances
    instances = outputs["instances"].to("cpu")
    return instances

def pixel_to_geo_polygon(tif_path, pixel_coords):
    # Open the TIFF file
    ds = gdal.Open(tif_path)

    # Get geotransformation information (affine transformation parameters)
    geotransform = ds.GetGeoTransform()

    x_min, y_min, x_max, y_max = pixel_coords

    # Convert pixel coordinates to geographic coordinates
    geo_x_min = geotransform[0] + x_min * geotransform[1] + y_min * geotransform[2]
    geo_y_min = geotransform[3] + x_min * geotransform[4] + y_min * geotransform[5]
    geo_x_max = geotransform[0] + x_max * geotransform[1] + y_max * geotransform[2]
    geo_y_max = geotransform[3] + x_max * geotransform[4] + y_max * geotransform[5]

    # Create a Shapely Polygon in geographic coordinates
    polygon = Polygon([(geo_x_min, geo_y_min), (geo_x_max, geo_y_min), (geo_x_max, geo_y_max), (geo_x_min, geo_y_max), (geo_x_min, geo_y_min)])
    
    return polygon

def generate_vector_bbox(in_files):
    
    # Create a list of geo coordinates and score
    geometries = list()
    sc = list()
    
    for files in os.listdir(in_files):
         if files.endswith(".tif"):
              file_path = os.path.join(in_files,files)
              
              detections =  visualizer(file_path)

              for i in range(len(detections)):
                detection = detections[i]
                class_id = detection.pred_classes.item()  # Get the class ID of the detection
                if class_id == 0:  # Assuming class one is indexed as 1
                    bbox = detection.pred_boxes.tensor[0].tolist()  # Get bounding box coordinates
                    score = detections.scores[i].item()

                    out = pixel_to_geo_polygon(file_path,bbox)
                    sc.append(score)
                    
                    # Add the bounding box information to the DataFrame
                    geometries.append(out)
                    

    return geometries, sc

def generate_shapefile(files):

    geo, score = generate_vector_bbox(files) 

    # Create a GeoSeries from the list of Shapely Polygon objects
    geometry_series = gpd.GeoSeries(geo, crs="EPSG:4326")

    # Create a GeoDataFrame from the GeoSeries
    gdf = gpd.GeoDataFrame(geometry_series, columns=['geometry'])

    # put scores in attribute table
    gdf['score'] = score

    output_shapefile = "merge.shp"

    # Save the GeoDataFrame as a shapefile
    gdf.to_file(output_shapefile)


tiles = r"C:\Users\User-1\Desktop\Data\detectron2\New folder\shahbaz_test\tiles"

# generate_shapefile(tiles)