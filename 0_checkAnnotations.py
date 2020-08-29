# 0_checkAnnotations.py
#
# This file checks the integrity of annotations on all .xml files which contain ROIs for H&E and TFF3 slides
# It creates so-called tile-dictionaries
#

import glob, os
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from shapely import geometry
import pyvips as pv
import numpy as np
import math
from tqdm import tqdm
import pickle, sys
sys.path.append('**path to pathml installation**') # install from here: https://github.com/9xg/pathml
from pathml import slide

whichStain = 'tff3' # he or tff3
tileSize=400 # extract 224 by 224 pixel tiles at given magnification level, previously we used 400 at 40x
magnificationLevel = 0 # 1 = 20(x) or 0 = 40(x)
enforceMagnificationLock = True # Only permit slides which were scanned at around 0.25 per micron
annotationCoeverageThreshold = {'he':0.33,'tff3':0.66}
# If the scan was successfully checked to be at 40x, all magnification coordinates have to the scaled by the downsample factor of the level difference (40x vs 20x)
magnificationLevelConverted = 40 if magnificationLevel==0 else 20


verbose = True
slidesRootFolder = "**Path to slides**"
heAnnotationsFileList = glob.glob("data/annotations/best2-"+whichStain+"-diag-annotations/BEST2_*.xml") # path to best2 annotations
if whichStain is 'he':
    classNames = ['Gastric-type columnar epithelium', 'Respiratory-type columnar epithelium', 'Background', 'Intestinal Metaplasia'] # Specify class names which at the same time are the names of the respective directory
elif whichStain is 'tff3':
    classNames = ['Equivocal', 'Positive', 'Negative','Stain']



for annotationFile in tqdm(heAnnotationsFileList,desc="Processing slides"):
    tileDictionary = {listKey:[] for listKey in classNames}
    slideFileName = os.path.split(annotationFile)[-1].replace(".xml",".svs")
    slideFolder = '_'.join(slideFileName.split("_")[0:3])

    print(slideFolder)
    #if os.path.isfile('data/tileDictionaries-tff3-300px-at-40x/'+slideFileName.replace('.svs','')+'.p'):
    if os.path.isfile('data/tileDictionaries-'+whichStain+'-'+str(tileSize)+'px-at-'+str(magnificationLevelConverted)+'x/'+slideFileName.replace('.svs','')+'.p'):
        print("Case already processed. Skipping...")
        continue
    heSlidePML = slide.Slide(slidesRootFolder + slideFolder + "/" + slideFileName,level=magnificationLevel)

    # Check whether the slide was scanned uniformly or whether there is any resolution corruption
    if float(heSlidePML.slideProperties['openslide.mpp-x']) != float(heSlidePML.slideProperties['openslide.mpp-y']):
        raise Warning('Mismatch between X and Y resolution (microns per pixel)')
    # Check whether the slides was scanned at 40x, otherwise fail
    if round(float(heSlidePML.slideProperties['openslide.mpp-x']),2) != 0.25 and enforceMagnificationLock:
        raise Warning('Slide not scanned at 40x')
    # Calculate the pixel size based on provided tile size and magnification level
    rescaledMicronsPerPixel = float(heSlidePML.slideProperties['openslide.mpp-x'])*float(heSlidePML.slideProperties['openslide.level['+str(magnificationLevel)+'].downsample'])
    #if verbose: print("Given the properties of this scan, the resulting tile size will correspond to "+str(round(tileSize*rescaledMicronsPerPixel,2))+" μm edge length")

    # Calculate scaling factor for annotations
    annotationScalingFactor = float(heSlidePML.slideProperties['openslide.level[0].downsample'])/float(heSlidePML.slideProperties['openslide.level['+str(magnificationLevel)+'].downsample'])

    print("Scale"+str(annotationScalingFactor))

    # Extract slide dimensions
    slideWidth = int(heSlidePML.slideProperties['width'])
    slideHeight = int(heSlidePML.slideProperties['height'])

    if verbose: print('Opening file: ' + annotationFile)
    tree = ET.parse(annotationFile) # Open .xml file
    root = tree.getroot() # Get root of .xml tree
    if root.tag == "ASAP_Annotations": # Check whether we actually deal with an ASAP .xml file
        if verbose: print('.xml file identified as ASAP annotation collection') # Display number of found annotations
    else:
        raise Warning('Not an ASAP .xml file')
    allHeAnnotations = root.find('Annotations') # Find all annotations for this slide
    if verbose: print('XML file valid - ' + str(len(allHeAnnotations)) + ' annotations found.') # Display number of found annotations

    # Generate a list of tile coordinates which we can extract
    for annotation in tqdm(allHeAnnotations,desc="Parsing annotations"):
        annotationTree = annotation.find('Coordinates')
        x = []
        y = []
        polygon = []
        for coordinate in annotationTree:
            info = coordinate.attrib
            polygon.append((float(info['X'])*annotationScalingFactor, float(info['Y'])*annotationScalingFactor))

        polygonNp = np.asarray(polygon)
        polygonNp[:,1] = slideHeight-polygonNp[:,1]
        poly = geometry.Polygon(polygonNp).buffer(0)

        topLeftCorner = (min(polygonNp[:,0]),max(polygonNp[:,1]))
        bottomRightCorner = (max(polygonNp[:,0]),min(polygonNp[:,1]))
        tilesInXax = math.ceil((bottomRightCorner[0] - topLeftCorner[0])/tileSize)
        tilesInYax = math.ceil((topLeftCorner[1] - bottomRightCorner[1])/tileSize)
        x = poly.exterior.coords.xy[0]
        y = poly.exterior.coords.xy[1]

        if poly.area >1.5*tileSize**2:
            for xTile in range(tilesInXax):
                for yTile in range(tilesInYax):
                    minX = topLeftCorner[0]+tileSize*xTile
                    minY = topLeftCorner[1]-tileSize*yTile
                    maxX = topLeftCorner[0]+tileSize*xTile+tileSize
                    maxY = topLeftCorner[1]-tileSize*yTile-tileSize
                    tileBox = geometry.box(minX,minY,maxX,maxY)

                    intersectingArea = poly.intersection(tileBox).area/tileSize**2
                    if intersectingArea>annotationCoeverageThreshold[whichStain]:
                        if annotation.attrib['PartOfGroup'] in tileDictionary:
                            tileDictionary[annotation.attrib['PartOfGroup']].append({'x': minX, 'y': slideHeight-minY, 'tileSize': tileSize, 'annotationCoverageArea': round(intersectingArea,4)})
                        else:
                            raise Warning('Annotation ' + annotation.attrib['PartOfGroup'] + ' is not part of a pre-defined group')
        else:
            minX = geometry.Point(poly.centroid).coords.xy[0][0]-tileSize/2
            minY = geometry.Point(poly.centroid).coords.xy[1][0]-tileSize/2
            maxX = geometry.Point(poly.centroid).coords.xy[0][0]+tileSize/2
            maxY = geometry.Point(poly.centroid).coords.xy[1][0]+tileSize/2
            tileBox = geometry.box(minX,minY,maxX,maxY)
            intersectingArea = poly.intersection(tileBox).area/tileSize**2

            if annotation.attrib['PartOfGroup'] in tileDictionary:
                tileDictionary[annotation.attrib['PartOfGroup']].append({'x': minX, 'y': slideHeight-minY-tileSize, 'tileSize': tileSize, 'annotationCoverageArea': intersectingArea})
            else:
                raise Warning('Annotation is not part of a pre-defined group')

    pickle.dump(tileDictionary, open('data/tileDictionaries-'+whichStain+'-'+str(tileSize)+'px-at-'+str(magnificationLevelConverted)+'x/'+slideFileName.replace('.svs','')+'.p', 'wb'))
