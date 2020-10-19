# 3-1_tff3PredictionsToDataframe.py
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

pathologyData = pd.read_excel('/media/gehrun01/work-io/cruk-phd-data/cytosponge/BEST2_DATA_EXTRACT_AUGUST2019/Views/BEST2_VW_PATHOLOGY_CRF_DATA_VIEW.xlsx')
endoscopyData = pd.read_excel('/media/gehrun01/work-io/cruk-phd-data/cytosponge/BEST2_DATA_EXTRACT_AUGUST2019/Views/BEST2_VW_ENDOSCOPY_CRF_DATA_VIEW.xlsx')
cytospongeData = pd.read_excel('/media/gehrun01/work-io/cruk-phd-data/cytosponge/BEST2_DATA_EXTRACT_AUGUST2019/cytospongeResults-clean.xlsx')

cytospongeData.dropna(subset=['Gland groups on HE','TFF3+'],inplace=True)
cytospongeData.replace({'Gland groups on HE': {'>5': 6}}, inplace=True)
endoscopyData.replace({'PRAGUE_M': {'NR': 0, '<1': 0.5}, 'PRAGUE_C': {'NR': 0, '<1': 0.5}}, inplace=True)
print('Run TFF3 predictions to dataframe conversion for ' + whichArchitecture)
rows_list = []
for case in tqdm(glob.glob('data/inferenceMaps/paper-tff3-' + whichArchitecture + '/*.p')):
    caseId = '_'.join(os.path.split(case)[-1].replace('.p', '').split('_')[0:3])
    caseIdSlash = '/'.join(os.path.split(case)[-1].replace('.p', '').split('_')[0:3])

    tileDictionary = pickle.load(open(case, "rb"))
    equivPredictionImage = np.zeros(tileDictionary['maskSize'][::-1])
    positivePredictionImage = np.zeros(tileDictionary['maskSize'][::-1])
    negativePredictionImage = np.zeros(tileDictionary['maskSize'][::-1])
    for key, val in tileDictionary['tileDictionary'].items():
        equivPredictionImage[key[1], key[0]] = val['prediction'][0] if 'prediction' in val else 0
        negativePredictionImage[key[1], key[0]] = val['prediction'][1] if 'prediction' in val else 0
        positivePredictionImage[key[1], key[0]] = val['prediction'][2] if 'prediction' in val else 0

    endoscopyGroundTruthEntry = endoscopyData.loc[(endoscopyData['PATIENT_ID'] == caseIdSlash) & (endoscopyData['VISIT'] == "Baseline")]
    cytospongeGroundTruthEntry = cytospongeData.loc[(cytospongeData['PATIENT ID'] == caseIdSlash)]
    pathologyGroundTruthEntry = pathologyData.loc[(pathologyData['PATIENT_ID'] == caseIdSlash) & (pathologyData['VISIT'] == "Baseline")]

    if len(pathologyGroundTruthEntry) is 0:
        print('Missing path data for ' + caseId)
        pathologistBiopsyCall = np.nan
        pathologistBiopsyResults = np.nan
    else:
        if pathologyGroundTruthEntry['HGD'].values[0] is np.nan:
            pathologistBiopsyCall = 0
            pathologistBiopsyResults = np.nan
        else:
            pathologistBiopsyCall = 1
            pathologistBiopsyResults = pathologyGroundTruthEntry['HGD'].values[0]

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

    rows_list.append([os.path.split(case)[-1].replace('.p', '')] + [np.count_nonzero(positivePredictionImage > prob) for prob in probabilitiesToThreshold] + [np.count_nonzero(equivPredictionImage > prob) for prob in probabilitiesToThreshold] + [cytospongePathologistCall, endoscopyC1M3Call, endoscopyC1M1Call, endoscopyC1Call, endoscopyC2Call, endoscopyC3Call, pathologistBiopsyCall, pathologistBiopsyResults])

tff3Predictions = pd.DataFrame(rows_list, columns=['Case'] + ['TFF3 positive count (> ' + str(round(prob, 6)) + ')' for prob in probabilitiesToThreshold] + ['TFF3 equivocal count (> ' + str(round(prob, 6)) + ')' for prob in probabilitiesToThreshold] + ['Cytosponge', 'Endoscopy (at least C1M3)', 'Endoscopy (at least C1M1)', 'Endoscopy (at least C1)', 'Endoscopy (at least C2)', 'Endoscopy (at least C3)', 'Pathologist biopsy (IM)', 'Pathologist biopsy (Result)'])
tff3Predictions['Endoscopy (at least C1 or M3) + Biopsy (IM)'] = (tff3Predictions['Endoscopy (at least C1M3)']==1) & (tff3Predictions['Pathologist biopsy (IM)']==1)
tff3Predictions.to_csv('data/slideLevelAggregation/TFF3-data-' + whichArchitecture + '.csv', index=False)
