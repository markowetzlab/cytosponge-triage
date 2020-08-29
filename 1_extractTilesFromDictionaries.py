# 1_extractTilesFromDictionaries.py
#
# This file checks takes tile-dictionaries and generates individual tiles as results
#

import glob
import os
import pickle

import numpy as np
import pyvips as pv
from tqdm import tqdm

verbose = True
magnificationLevel = 0  # 1 = 20(x) or 0 = 40(x)
slidesRootFolder = "**Path to slides**"
tileDictionariesFileList = glob.glob(
    "**Path to tile dictionaries**")
tilesRootFolder = "**Path to tiles folder**"

format_to_dtype = {
    'uchar': np.uint8,
    'char': np.int8,
    'ushort': np.uint16,
    'short': np.int16,
    'uint': np.uint32,
    'int': np.int32,
    'float': np.float32,
    'double': np.float64,
    'complex': np.complex64,
    'dpcomplex': np.complex128,
}

for tileDictionaryFile in tqdm(tileDictionariesFileList, desc="Processing tile dictionaries"):
    if verbose:
        print('Opening file: ' + tileDictionaryFile)
    tileDictionary = pickle.load(open(tileDictionaryFile, "rb"))
    slideFileName = os.path.split(tileDictionaryFile)[-1].replace(".p", ".svs")
    slideFolder = '_'.join(slideFileName.split("_")[0:3])
    heSlide = pv.Image.new_from_file(
        slidesRootFolder + slideFolder + "/" + slideFileName, level=magnificationLevel)
    try:
        os.mkdir(tilesRootFolder + slideFolder)
        print("Directory ", tilesRootFolder + slideFolder, " Created ")
    except FileExistsError:
        print("Directory ", tilesRootFolder + slideFolder, " already exists")
        continue
    for key in tileDictionary:
        try:
            os.mkdir(tilesRootFolder + slideFolder + "/" + key)
            print("Directory ", tilesRootFolder + slideFolder + "/" + key, " Created ")
        except FileExistsError:
            print("Directory ", tilesRootFolder + slideFolder + "/" + key, " already exists")
        # Iterate over all tiles, extract them and save
        for tile in tqdm(tileDictionary[key]):
            if not os.path.exists(tilesRootFolder + slideFolder + "/" + key + '/' + slideFolder + '_' + str(tile['x']) + '_' + str(tile['y']) + '_' + str(tile['tileSize']) + '.jpg'):
                areaHe = heSlide.extract_area(
                    tile['x'], tile['y'], tile['tileSize'], tile['tileSize'])
                areaHe.write_to_file(tilesRootFolder + slideFolder + "/" + key + '/' + slideFolder + '_' + str(
                    tile['x']) + '_' + str(tile['y']) + '_' + str(tile['tileSize']) + '.jpg', Q=100)
