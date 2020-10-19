# 3_runInferenceOnSlides.py
#
# This file runs tile-level inference on the training, calibration and internal validation cohort
#


import sys
import time
import numpy as np
import pyvips as pv
import warnings
from torchvision import datasets, models, transforms
import torch
from torch.autograd import Variable
from PIL import Image
import imageio
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import os
import argparse
from pathlib import Path
from WholeSlideImageDataset import WholeSlideImageDataset

sys.path.append('**path to pathml installation**') # install from here: https://github.com/9xg/pathml
from pathml import slide

parser = argparse.ArgumentParser(description='Run inference on slides.')
parser.add_argument("-stain", required=True, help="he or tff3")
parser.add_argument("-model", required=True, help="model file")
parser.add_argument("-inferencefolder", required=True, help="inference folder")
parser.add_argument("-architecturetilesize", required=True, help="architecture tile size")
parser.add_argument("-foregroundOnly", required=True, help="foreground with tissue only")
parser.add_argument("-slidesRootFolder", required=True, help="slider root folder")
args = parser.parse_args()
print(args)

trainedModel = torch.load('model_files/torch-models/' + args.model)
inferenceMapFolder = args.inferencefolder
whichStain = args.stain
pathTileSize = 400
batchSizeForInference = 30
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
trainedModel.to(device)
trainedModel.eval()

patch_size = int(args.architecturetilesize)

data_transforms = transforms.Compose([
    transforms.Resize(patch_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


slidesRootFolder = args.slidesRootFolder
caseList = Path(slidesRootFolder).rglob('*' + whichStain.upper() + '*.svs')

if whichStain == 'tff3':
    heAnnotationsList = ['_'.join(os.path.split(i)[-1].split('_')[0:5]).replace('.xml', '') for i in glob.glob("data/annotations/best2-" + whichStain + "-diag-annotations/*.xml")]
elif whichStain == 'he':
    heAnnotationsList = ['_'.join(os.path.split(i)[-1].split('_')[0:5]).replace('.xml', '') for i in glob.glob("data/annotations/best2-" + whichStain + "-qc-annotations/*.xml")]
#heAnnotationsList = []


for caseFolder in tqdm(caseList):
    caseFolder = str(caseFolder)
    caseName = os.path.split(caseFolder)[-1].replace('.svs', '')
    print(caseName)
    #print(caseFolder)
    if caseName in heAnnotationsList:
        if os.path.isfile('data/inferenceMaps/' + inferenceMapFolder + '/' + caseName + '.p'):
            os.remove('data/inferenceMaps/' + inferenceMapFolder + '/' + caseName + '.p')
        print("This case was included in the training set. Skipping...")
        continue
    if os.path.isfile('data/inferenceMaps/' + inferenceMapFolder + '/' + caseName + '.p'):
        print("Case already processed. Skipping...")
        continue

    pathSlide = slide.Slide(caseFolder)

    pathSlide.setTileProperties(tileSize=pathTileSize)
    # , tileOverlap=0.33
    pathSlide.detectForeground(threshold=95)

    pathSlideDataset = WholeSlideImageDataset(
        pathSlide, foregroundOnly=True, transform=data_transforms)

    since = time.time()
    pathSlideDataloader = torch.utils.data.DataLoader(pathSlideDataset, batch_size=batchSizeForInference, shuffle=False, num_workers=16)
    for inputs in tqdm(pathSlideDataloader):
        inputTile = inputs['image'].to(device)
        output = trainedModel(inputTile)
        output = output.to(device)

        batch_prediction = torch.nn.functional.softmax(
            output, dim=1).cpu().data.numpy()

        # Reshape it is a Todo - instead of for looping
        for index in range(len(inputTile)):
            tileAddress = (inputs['tileAddress'][0][index].item(),
                           inputs['tileAddress'][1][index].item())
            pathSlide.appendTag(tileAddress, 'prediction', batch_prediction[index, ...])
    tileDictionaryWithInference = {'maskSize': (
        pathSlide.numTilesInX, pathSlide.numTilesInY), 'tileDictionary': pathSlide.tileMetadata}
    pickle.dump(tileDictionaryWithInference, open(
        'data/inferenceMaps/' + inferenceMapFolder + '/' + caseName + '.p', 'wb'))
    time_elapsed = time.time() - since
    print('Inference complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
