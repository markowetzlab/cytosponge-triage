# 4-1_fullyAutomatedModelCalibration.py
#
# This script calibrates the fully automated model
#

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.utils import resample
from sklearn.metrics import auc, confusion_matrix, roc_auc_score, roc_curve
import pickle
plt.style.use('figures/journal-style.mplstyle')


calibrationCohort = pickle.load( open( "data/slideLevelAggregation/calibrationCohort.p", "rb" ) )
calibrationCohortHE = [s + '_HE_1' for s in calibrationCohort]
calibrationCohortTFF3 = [s + '_TFF3_1' for s in calibrationCohort]

csvFilesToPlot = ['HE-data-alexnet.csv', 'HE-data-densenet.csv', 'HE-data-inceptionv3.csv',
                  'HE-data-resnet18.csv', 'HE-data-squeezenet.csv', 'HE-data-vgg16.csv']
architectureNames = ['AlexNet', 'Densenet',
                     'Inception v3', 'ResNet-18', 'SqueezeNet', 'VGG-16']

colorLUT = {'AlexNet': [.58,.404,.741], 'Densenet': [1,.498,.055], 'Inception v3': [.122,.467,.706], 'ResNet-18': [.839,.153,.157], 'SqueezeNet': [.220,.647,.220], 'VGG-16': [.549,.337,.294]}

# Gastric columns : [205:-211]
columnTitle = 'Gastric count (> X)'

bestProbabilitiesForCutoff = []

probabilitiesToThreshold = np.append(np.arange(0, 1, 0.005),[0.999,0.9999,0.99999])

thresholdList = []

heCutoffs = {key:{} for key in csvFilesToPlot}
probAucLists = {key:{} for key in csvFilesToPlot}
for csvFile in csvFilesToPlot:
    aucList = []
    fprList = []
    tprList = []
    heData = pd.read_csv('data/slideLevelAggregation/' + csvFile)
    heData = heData[heData['Case'].isin(calibrationCohortHE)]
    for column in heData.columns[204:-210]:
        columnAuc = roc_auc_score(
            heData['Cytosponge QC'], heData[column])
        aucList.append(columnAuc)
        fpr, tpr, thresholds = roc_curve(
            heData['Cytosponge QC'], heData[column])
        fprList.append(fpr)
        tprList.append(tpr)

    m = max(aucList)
    mIdx = aucList.index(m)
    aucListMax = [i for i, j in enumerate(aucList) if j == m]
    print(csvFile, 'Probability: ' +
          str(probabilitiesToThreshold[aucListMax[0]]), 'AUC: ' + str(aucList[aucListMax[0]]))
    bestProbabilitiesForCutoff.append(round(probabilitiesToThreshold[aucListMax[0]],6))
    heCutoffs[csvFile]['tileThreshold'] = probabilitiesToThreshold[aucListMax[0]]
    probAucLists[csvFile]['prob'] = aucList

print(heCutoffs)
with open('data/slideLevelAggregation/heThresholds.p', 'wb') as f:
    pickle.dump(heCutoffs, f)

print('-'*50)

plt.figure(figsize=(7.5,6))
for csvIdx,csvFile in enumerate(csvFilesToPlot):
    plt.plot(probabilitiesToThreshold,probAucLists[csvFile]['prob'],c=colorLUT[architectureNames[csvIdx]],label=architectureNames[csvIdx])
plt.legend(loc='lower center', prop={'size': 12})
plt.xlim(-0.05, 1.05)
plt.ylim(0.6, 1.0)
plt.xlabel('Probability threshold for determination\nof number of tiles with columnar epithlium')
plt.ylabel('AUC-ROC for Cytosponge QC with\nthresholded number of tiles')
plt.savefig('figures/architectureComparison/probThreshold-HE.pdf')
plt.show()


csvFilesToPlot = ['TFF3-data-alexnet.csv', 'TFF3-data-densenet.csv', 'TFF3-data-inceptionv3.csv',
                  'TFF3-data-resnet18.csv', 'TFF3-data-squeezenet.csv', 'TFF3-data-vgg16.csv']
architectureNames = ['AlexNet', 'Densenet',
                     'Inception v3', 'ResNet-18', 'SqueezeNet', 'VGG-16']

# uncomment for main paper figure
#csvFilesToPlot = ['TFF3-data-inceptionv3.csv', 'TFF3-data-squeezenet.csv']
#architectureNames = ['Inception v3', 'SqueezeNet']
# Positive columns : tff3Data.columns[1:-213]
# Equivocal columns : tff3Data.columns[201:-2]
columnTitle = 'TFF3 positive count (> X)'

bestProbabilitiesForCutoff = []

probabilitiesToThreshold = np.append(np.arange(0, 1, 0.005),[0.999,0.9999,0.99999])

tff3Cutoffs = {key:{} for key in csvFilesToPlot}
probAucLists = {key:{} for key in csvFilesToPlot}
for csvFile in csvFilesToPlot:
    aucList = []
    fprList = []
    tprList = []
    thresholdList = []
    tff3Data = pd.read_csv('data/slideLevelAggregation/' + csvFile)
    tff3Data = tff3Data[tff3Data['Case'].isin(calibrationCohortTFF3)]

    for column in tff3Data.columns[1:-212]:
        columnAuc = roc_auc_score(tff3Data['Endoscopy (at least C1 or M3) + Biopsy (IM)'], tff3Data[column])
        aucList.append(columnAuc)
        fpr, tpr, thresholds = roc_curve(tff3Data['Endoscopy (at least C1 or M3) + Biopsy (IM)'], tff3Data[column],drop_intermediate=False)
        fprList.append(fpr)
        tprList.append(tpr)
        thresholdList.append(thresholds)

    m = max(aucList)
    aucListMax = [i for i, j in enumerate(aucList) if j == m]
    print(csvFile, 'Probability: ' +
          str(probabilitiesToThreshold[aucListMax[0]]), 'AUC: ' + str(aucList[aucListMax[0]]))
    bestProbabilitiesForCutoff.append(probabilitiesToThreshold[aucListMax[0]])
    tff3Cutoffs[csvFile]['tileThreshold'] = probabilitiesToThreshold[aucListMax[0]]


    tn, fp, fn, tp = confusion_matrix(tff3Data['Endoscopy (at least C1 or M3) + Biopsy (IM)'], tff3Data['Cytosponge']).ravel()
    sensitivityPath = tp/(tp+fn)
    specificityPath = tn/(tn+fp)

    specificityFpr = min(fprList[aucListMax[0]], key=lambda x:abs(x-(1-specificityPath)))
    #quit()
    indexForTpr = np.where(fprList[aucListMax[0]] == specificityFpr)
    print('Sensitivity when fixing specificity at '+str(specificityPath*100)+'%: '+ str(tprList[aucListMax[0]][indexForTpr[0][0]]) +'% at threshold of '+str(thresholdList[aucListMax[0]][indexForTpr[0][0]]))
    tff3Cutoffs[csvFile]['slideThreshold'] = thresholdList[aucListMax[0]][indexForTpr[0][0]]
    probAucLists[csvFile]['prob'] = aucList
print(tff3Cutoffs)
plt.figure(figsize=(7.5,6))
for csvIdx,csvFile in enumerate(csvFilesToPlot):
    plt.plot(probabilitiesToThreshold,probAucLists[csvFile]['prob'],c=colorLUT[architectureNames[csvIdx]],label=architectureNames[csvIdx])
plt.legend(loc='lower center', prop={'size': 12})
plt.xlim(-0.05, 1.05)
plt.ylim(0.6, 1.0)
plt.xlabel('Probability threshold for determination\nof number of tiles with positive goblet cells')
plt.ylabel('AUC-ROC for BE on endoscopy (+ IM) with\nthresholded number of tiles')
plt.savefig('figures/architectureComparison/probThreshold-TFF3.pdf')

plt.show()

with open('data/slideLevelAggregation/tff3Thresholds.p', 'wb') as f:
    pickle.dump(tff3Cutoffs, f)
