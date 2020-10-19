import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import pickle
import argparse
from sklearn.utils import resample
from sklearn.metrics import auc, confusion_matrix, roc_auc_score, roc_curve
calibrationCohort = pickle.load( open( "data/slideLevelAggregation/calibrationCohort.p", "rb" ) )
calibrationCohortTFF3 = [s + '_TFF3_1' for s in calibrationCohort]

parser = argparse.ArgumentParser(description='Run inference on slides.')
parser.add_argument("-mode", required=True, help="Plot calibration or validation cohort")
parser.add_argument("-suppl", required=True, help="Plot all architectures or just best/worst")
parser.add_argument("-ci", required=True, help="Plot CIs")
args = parser.parse_args()
whichCohortSubset = args.mode
supplMode = int(args.suppl)
plotCI = int(args.ci)

plt.style.use('figures/journal-style.mplstyle')


if supplMode == 1:
    csvFilesToPlot = ['TFF3-data-alexnet.csv', 'TFF3-data-densenet.csv', 'TFF3-data-inceptionv3.csv',
                      'TFF3-data-resnet18.csv', 'TFF3-data-squeezenet.csv', 'TFF3-data-vgg16.csv']
    architectureNames = ['AlexNet', 'Densenet',
                         'Inception v3', 'ResNet-18', 'SqueezeNet', 'VGG-16']
elif supplMode == 0:
    csvFilesToPlot = ['TFF3-data-squeezenet.csv', 'TFF3-data-vgg16.csv']
    architectureNames = ['SqueezeNet', 'VGG-16']


colorLUT = {'AlexNet': [.58,.404,.741], 'Densenet': [1,.498,.055], 'Inception v3': [.122,.467,.706], 'ResNet-18': [.839,.153,.157], 'SqueezeNet': [.220,.647,.220], 'VGG-16': [.549,.337,.294]}

columnTitle = 'TFF3 positive count (> X)'

with open('data/slideLevelAggregation/tff3Thresholds.p','rb') as f:
    tff3Cutoffs = pickle.load(f)

plotTitles = ['TFF3 positive tiles vs. endoscopy (C1M3+IM)', 'TFF3 positive tiles vs. pathologist']
groundTruthComp = ['Endoscopy (at least C1 or M3) + Biopsy (IM)', 'Cytosponge']
figureFileNames = ['TFF3-IM-Endoscopy(C1M3)_'+whichCohortSubset+'.pdf', 'TFF3-IM-Cytosponge(TFF3+)_'+whichCohortSubset+'.pdf']

for groundTruth, plotTitle, figureFileName in zip(groundTruthComp, plotTitles, figureFileNames):
    plt.figure(figsize=(6, 6))
    for csvIdx, csvFile in enumerate(csvFilesToPlot):
        aucList = []
        fprList = []
        tprList = []
        tff3Data = pd.read_csv('data/slideLevelAggregation/' + csvFile)
        if whichCohortSubset == 'validation':
            tff3Data = tff3Data[~tff3Data['Case'].isin(calibrationCohortTFF3)]
        elif whichCohortSubset == 'calibration':
            tff3Data = tff3Data[tff3Data['Case'].isin(calibrationCohortTFF3)]

        # configure bootstrap
        n_iterations = 500
        n_size = int(len(tff3Data) * 1)
        lw = 2

        bootstrapAucList = []
        bootstrapTprList = []
        bootstrapFprList = []
        bootstrapThresholdList = []
        bootstrapSensitivityList = []
        bootstrapSpecificityList = []
        bootstrapPathSensitivityList = []
        bootstrapPathSpecificityList = []

        interpolatedBootstrapTprList = []
        toProbe = np.linspace(0, 1, num=101)

        for i in range(n_iterations):
            bootstrapTff3Data = resample(tff3Data, n_samples=n_size)
            #print(train)
            fpr, tpr, thresholds = roc_curve(bootstrapTff3Data[groundTruth], bootstrapTff3Data[columnTitle.replace(
                'X', str(tff3Cutoffs[csvFile]['tileThreshold']))],drop_intermediate=False)
            roc_auc = auc(fpr, tpr)

            tn, fp, fn, tp = confusion_matrix(bootstrapTff3Data[groundTruth], bootstrapTff3Data[columnTitle.replace(
                'X', str(tff3Cutoffs[csvFile]['tileThreshold']))]>=tff3Cutoffs[csvFile]['slideThreshold']).ravel()
            bootstrapSensitivityList.append(tp/(tp+fn))
            bootstrapSpecificityList.append(tn/(tn+fp))
            bootstrapAucList.append(roc_auc)
            bootstrapTprList.append(tpr)
            bootstrapFprList.append(fpr)
            bootstrapThresholdList.append(thresholds)
            interpolatedBootstrapTprList.append(np.interp(toProbe,fpr,tpr))

            tn, fp, fn, tp = confusion_matrix(bootstrapTff3Data[groundTruth], bootstrapTff3Data['Cytosponge']).ravel()
            bootstrapPathSensitivityList.append(tp/(tp+fn))
            bootstrapPathSpecificityList.append(tn/(tn+fp))
              #test = numpy.array([x for x in values if x.tolist() not in train.tolist()])
        # plot scores
        #plt.hist(bootstrapAucList)
        #plt.show()
        # confidence intervals
        alpha = 0.95
        p = ((1.0-alpha)/2.0) * 100
        lowerAuc = max(0.0, np.percentile(bootstrapAucList, p))
        lowerSens = max(0.0, np.percentile(bootstrapSensitivityList, p))
        lowerSpec = max(0.0, np.percentile(bootstrapSpecificityList, p))
        lowerPathSens = max(0.0, np.percentile(bootstrapPathSensitivityList, p))
        lowerPathSpec = max(0.0, np.percentile(bootstrapPathSpecificityList, p))
        p = (alpha+((1.0-alpha)/2.0)) * 100
        upperAuc = min(1.0, np.percentile(bootstrapAucList, p))
        upperSens = min(1.0, np.percentile(bootstrapSensitivityList, p))
        upperSpec = min(1.0, np.percentile(bootstrapSpecificityList, p))
        upperPathSens = min(1.0, np.percentile(bootstrapPathSensitivityList, p))
        upperPathSpec = min(1.0, np.percentile(bootstrapPathSpecificityList, p))

        lowerCICurve=[]
        upperCICurve=[]
        for fprIndex, fprProbe in np.ndenumerate(toProbe):
            thisRun = []
            for bootstrapRuns in interpolatedBootstrapTprList:
                thisRun.append(bootstrapRuns[fprIndex])
            p = ((1.0-alpha)/2.0) * 100
            lowerCICurve.append(max(0.0, np.percentile(thisRun, p)))
            p = (alpha+((1.0-alpha)/2.0)) * 100
            upperCICurve.append(min(1.0, np.percentile(thisRun, p)))

        fpr, tpr, thresholds = roc_curve(tff3Data[groundTruth], tff3Data[columnTitle.replace(
            'X', str(tff3Cutoffs[csvFile]['tileThreshold']))], drop_intermediate=False)

        if groundTruth == "Endoscopy (at least C1 or M3) + Biopsy (IM)":
            tn, fp, fn, tp = confusion_matrix(tff3Data[groundTruth], tff3Data[columnTitle.replace(
                'X', str(tff3Cutoffs[csvFile]['tileThreshold']))]>=tff3Cutoffs[csvFile]['slideThreshold']).ravel()
            print('-'*20)
            print(architectureNames[csvIdx]+':')
            print('Sensitivity with op from calib: %0.3f%% (%0.3f%% - %0.3f%%)' % (tp/(tp+fn)*100,lowerSens*100,upperSens*100))
            print('Specificity with op from calib: %0.3f%% (%0.3f%% - %0.3f%%)' % (tn/(tn+fp)*100,lowerSpec*100,upperSpec*100))


        if whichCohortSubset == 'validation' and groundTruth == "Endoscopy (at least C1 or M3) + Biopsy (IM)":
            threshold = min(thresholds, key=lambda x:abs(x-tff3Cutoffs[csvFile]['slideThreshold']))
            indexForThreshold = np.where(thresholds == threshold)
            plt.plot(fpr[indexForThreshold], tpr[indexForThreshold], marker='o', markersize=8, color=colorLUT[architectureNames[csvIdx]],label=architectureNames[csvIdx] + ' calibration point',linewidth=0)
            plt.axhline(y=tpr[indexForThreshold],xmin=0,xmax=fpr[indexForThreshold],linestyle='--',color=colorLUT[architectureNames[csvIdx]],alpha=0.4)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr,
                 lw=lw, label=architectureNames[csvIdx] + ' - AUC: %0.2f (CI: %0.2f-%0.2f)' % (roc_auc, lowerAuc, upperAuc), color=colorLUT[architectureNames[csvIdx]])
        if plotCI == 1: plt.fill_between(toProbe,lowerCICurve,upperCICurve,color=colorLUT[architectureNames[csvIdx]],alpha=0.25,lw=0)

        #plt.plot(fpr, tpr,
        #         lw=lw, label=architectureNames[csvIdx] + ' | AUC = %0.3f (CI 95%%: %0.3f - %0.3f / Sens at 90%% spec: %0.2f' % (roc_auc, lower, upper, round(tpr[indexForTpr[0][0]]*100,2)))
        #plt.plot(fpr, tpr,
        #         lw=lw, label=architectureNames[csvIdx] + ' | AUC = %0.2f' % (roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    if groundTruth == "Endoscopy (at least C1 or M3) + Biopsy (IM)":
        tn, fp, fn, tp = confusion_matrix(tff3Data[groundTruth], tff3Data['Cytosponge']).ravel()
        sensitivityPath = tp/(tp+fn)
        specificityPath = tn/(tn+fp)
        plt.plot([1 - specificityPath], [sensitivityPath], marker='o', markersize=8, color="red",label="Pathologist",linewidth=0)
        plt.axhline(y=sensitivityPath,xmin=0,xmax=1-specificityPath,linestyle='--',color='red',alpha=0.4)
        plt.axvline(x=1-specificityPath,ymin=0,ymax=sensitivityPath-0.04,linestyle='--',color='red',alpha=0.4)
        print('-'*20)
        print('Pathologist:')
        print('Sensitivity: %0.3f%% (%0.3f%% - %0.3f%%)' % (sensitivityPath*100,lowerPathSens*100,upperPathSens*100))
        print('Specificity: %0.3f%% (%0.3f%% - %0.3f%%)' % (specificityPath*100,lowerPathSpec*100,upperPathSpec*100))


    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    #plt.title(plotTitle)
    plt.legend(loc="lower right", prop={'size': 12})
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.gca().legend(handles[::-1], labels[::-1])
    plt.show(block=False)
    plt.tight_layout()
    plt.savefig('figures/architectureComparison/' + figureFileName)

plt.show()
