# 3-1_hePredictionsToDataframe.py
#
# This script converts tile predictions of individual whole-slide images to aggregation dataframes
#

import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import imageio
from skimage.filters import gaussian
from tqdm import tqdm
import argparse
import glob
import os

parser = argparse.ArgumentParser(description='Convert predictions into differential counts.')
parser.add_argument("-architecture", required=True, help="CNN architecture")
args = parser.parse_args()

whichArchitecture = args.architecture

probabilitiesToThreshold = np.append(np.arange(0, 1, 0.005),[0.999,0.9999,0.99999])

endoscopyData = pd.read_excel('/media/gehrun01/work-io/cruk-phd-data/cytosponge/BEST2_DATA_EXTRACT_AUGUST2019/Views/BEST2_VW_ENDOSCOPY_CRF_DATA_VIEW.xlsx')
cytospongeData = pd.read_excel('/media/gehrun01/work-io/cruk-phd-data/cytosponge/BEST2_DATA_EXTRACT_AUGUST2019/cytospongeResults-clean.xlsx')

cytospongeData.dropna(subset=['Gland groups on HE','TFF3+'],inplace=True)
cytospongeData.replace({'Gland groups on HE': {'>5': 6}}, inplace=True)
endoscopyData.replace({'PRAGUE_M': {'NR': 0, '<1': 0.5}, 'PRAGUE_C': {'NR': 0, '<1': 0.5}}, inplace=True)
print('Run H&E predictions to dataframe conversion for ' + whichArchitecture)
rows_list = []
for case in tqdm(glob.glob('data/inferenceMaps/paper-he-' + whichArchitecture + '/*.p')):
    caseId = '_'.join(os.path.split(case)[-1].replace('.p', '').split('_')[0:3])
    caseIdSlash = '/'.join(os.path.split(case)[-1].replace('.p', '').split('_')[0:3])

    tileDictionary = pickle.load(open(case, "rb"))
    respPredictionImage = np.zeros(tileDictionary['maskSize'][::-1])
    gastPredictionImage = np.zeros(tileDictionary['maskSize'][::-1])
    imPredictionImage = np.zeros(tileDictionary['maskSize'][::-1])
    for key, val in tileDictionary['tileDictionary'].items():
        respPredictionImage[key[1], key[0]] = val['prediction'][3] if 'prediction' in val else 0
        gastPredictionImage[key[1], key[0]] = val['prediction'][1] if 'prediction' in val else 0
        imPredictionImage[key[1], key[0]] = val['prediction'][2] if 'prediction' in val else 0

    endoscopyGroundTruthEntry = endoscopyData.loc[(endoscopyData['PATIENT_ID'] == caseIdSlash) & (endoscopyData['VISIT'] == "Baseline")]
    cytospongeGroundTruthEntry = cytospongeData.loc[(cytospongeData['PATIENT ID'] == caseIdSlash)]

    if len(endoscopyGroundTruthEntry) is 0 or len(cytospongeGroundTruthEntry) is 0:
        print('Missing data for ' + caseId)
        continue

    if int(cytospongeGroundTruthEntry['Gland groups on HE'].values[0]) > 4:
        cytospongePathologistQCCall = 1
    else:
        cytospongePathologistQCCall = 0

    cytospongePathologistCall = int(cytospongeGroundTruthEntry['TFF3+'].values[0])

    if endoscopyGroundTruthEntry['PRAGUE_C'].values[0] == "CA":
        print('Missing data for ' + caseId)
        continue

    if int(endoscopyGroundTruthEntry['PRAGUE_C'].values[0]) >= 1 or int(endoscopyGroundTruthEntry['PRAGUE_M'].values[0]) >= 3:
        endoscopyC1M3Call = 1
    else:
        endoscopyC1M3Call = 0

    if int(endoscopyGroundTruthEntry['PRAGUE_C'].values[0]) >= 1 or int(endoscopyGroundTruthEntry['PRAGUE_M'].values[0]) >= 1:
        endoscopyC1M1Call = 1
    else:
        endoscopyC1M1Call = 0

    if int(endoscopyGroundTruthEntry['PRAGUE_C'].values[0]) >= 1:
        endoscopyC1Call = 1
    else:
        endoscopyC1Call = 0

    if int(endoscopyGroundTruthEntry['PRAGUE_C'].values[0]) >= 2:
        endoscopyC2Call = 1
    else:
        endoscopyC2Call = 0

    if int(endoscopyGroundTruthEntry['PRAGUE_C'].values[0]) >= 3:
        endoscopyC3Call = 1
    else:
        endoscopyC3Call = 0

    rows_list.append([os.path.split(case)[-1].replace('.p', '')] + [np.count_nonzero(imPredictionImage > prob) for prob in probabilitiesToThreshold] + [np.count_nonzero(gastPredictionImage > prob) for prob in probabilitiesToThreshold] + [np.count_nonzero(respPredictionImage > prob) for prob in probabilitiesToThreshold] + [cytospongePathologistQCCall, cytospongePathologistCall, endoscopyC1M3Call, endoscopyC1M1Call, endoscopyC1Call, endoscopyC2Call, endoscopyC3Call])

hePredictions = pd.DataFrame(rows_list, columns=['Case'] + ['IM positive count (> ' + str(round(prob, 6)) + ')' for prob in probabilitiesToThreshold] + ['Gastric count (> ' + str(round(prob, 6)) + ')' for prob in probabilitiesToThreshold] + ['Resp count (> ' + str(round(prob, 6)) + ')' for prob in probabilitiesToThreshold] + ['Cytosponge QC', 'Cytosponge TFF3+', 'Endoscopy (at least C1M3)', 'Endoscopy (at least C1M1)', 'Endoscopy (at least C1)', 'Endoscopy (at least C2)', 'Endoscopy (at least C3)'])
hePredictions.to_csv('data/slideLevelAggregation/HE-data-' + whichArchitecture + '.csv', index=False)
