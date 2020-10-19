import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import pickle
import argparse
from sklearn.utils import resample
from sklearn.metrics import auc, confusion_matrix, roc_auc_score, roc_curve
calibrationCohort = pickle.load( open( "data/slideLevelAggregation/calibrationCohort.p", "rb" ) )
calibrationCohortHE = [s + '_HE_1' for s in calibrationCohort]

plt.style.use('figures/journal-style.mplstyle')

parser = argparse.ArgumentParser(description='Run inference on slides.')
parser.add_argument("-mode", required=True, help="Plot calibration or validation cohort")
parser.add_argument("-suppl", required=True, help="Plot all architectures or just best/worst")
parser.add_argument("-ci", required=True, help="Plot CIs")
args = parser.parse_args()
whichCohortSubset = args.mode
supplMode = int(args.suppl)
plotCI = int(args.ci)

if supplMode == 1:
    csvFilesToPlot = ['HE-data-alexnet.csv', 'HE-data-densenet.csv', 'HE-data-inceptionv3.csv',
                      'HE-data-resnet18.csv', 'HE-data-squeezenet.csv', 'HE-data-vgg16.csv']
    architectureNames = ['AlexNet', 'Densenet',
                         'Inception v3', 'ResNet-18', 'SqueezeNet', 'VGG-16']
elif supplMode == 0:
    csvFilesToPlot = ['HE-data-squeezenet.csv', 'HE-data-vgg16.csv']
    architectureNames = ['SqueezeNet', 'VGG-16']

# uncomment for main paper figure
#csvFilesToPlot = ['HE-data-squeezenet.csv', 'HE-data-vgg16.csv']
#architectureNames = ['SqueezeNet', 'VGG-16']

colorLUT = {'AlexNet': [.58,.404,.741], 'Densenet': [1,.498,.055], 'Inception v3': [.122,.467,.706], 'ResNet-18': [.839,.153,.157], 'SqueezeNet': [.220,.647,.220], 'VGG-16': [.549,.337,.294]}


# Gastric columns : columns[201:-203]
columnTitle = 'Gastric count (> X)'

with open('data/slideLevelAggregation/heThresholds.p','rb') as f:
    heCutoffs = pickle.load(f)

plotTitles = ['H&E columnar cells vs. pathologist']
groundTruthComp = ['Cytosponge QC']
figureFileNames = ['HE-QC-Pathologist.pdf']

for groundTruth, plotTitle, figureFileName in zip(groundTruthComp, plotTitles, figureFileNames):
    plt.figure(figsize=(6, 6))
    for csvIdx, csvFile in enumerate(csvFilesToPlot):
        aucList = []
        fprList = []
        tprList = []
        heData = pd.read_csv('data/slideLevelAggregation/' + csvFile)
        if whichCohortSubset == 'validation':
            heData = heData[~heData['Case'].isin(calibrationCohortHE)]
        elif whichCohortSubset == 'calibration':
            heData = heData[heData['Case'].isin(calibrationCohortHE)]

        # configure bootstrap
        n_iterations = 500
        n_size = int(len(heData) * 1)

        lw = 2

        bootstrapAucList = []
        bootstrapTprList = []
        bootstrapFprList = []
        bootstrapThresholdList = []
        interpolatedBootstrapTprList=[]
        toProbe = np.linspace(0, 1, num=101)
        for i in range(n_iterations):
            bootstrapHeData = resample(heData, n_samples=n_size)
            #print(train)
            fpr, tpr, thresholds = roc_curve(bootstrapHeData[groundTruth], bootstrapHeData[columnTitle.replace(
                'X', str(heCutoffs[csvFile]['tileThreshold']))])
            roc_auc = auc(fpr, tpr)

            bootstrapAucList.append(roc_auc)
            bootstrapTprList.append(tpr)
            bootstrapFprList.append(fpr)
            bootstrapThresholdList.append(thresholds)
            interpolatedBootstrapTprList.append(np.interp(toProbe,fpr,tpr))

              #test = numpy.array([x for x in values if x.tolist() not in train.tolist()])

        # plot scores
        #plt.hist(bootstrapAucList)
        #plt.show()
        # confidence intervals
        alpha = 0.95
        p = ((1.0-alpha)/2.0) * 100
        lower = max(0.0, np.percentile(bootstrapAucList, p))
        p = (alpha+((1.0-alpha)/2.0)) * 100
        upper = min(1.0, np.percentile(bootstrapAucList, p))

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


        fpr, tpr, thresholds = roc_curve(heData[groundTruth], heData[columnTitle.replace(
            'X', str(heCutoffs[csvFile]['tileThreshold']))])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr,
                 lw=lw, label=architectureNames[csvIdx] + ' - AUC: %0.2f (CI: %0.2f-%0.2f)' % (roc_auc, lower, upper), color=colorLUT[architectureNames[csvIdx]])
        if plotCI == 1: plt.fill_between(toProbe,lowerCICurve,upperCICurve,color=colorLUT[architectureNames[csvIdx]],alpha=0.25,lw=0)
        #plt.plot(fpr, tpr,
        #         lw=lw, label=architectureNames[csvIdx] + ' | AUC = %0.3f (CI 95%%: %0.3f - %0.3f / Sens at 90%% spec: %0.2f' % (roc_auc, lower, upper, round(tpr[indexForTpr[0][0]]*100,2)))
        #plt.plot(fpr, tpr,
    #                     lw=lw, label=architectureNames[csvIdx] + ' | AUC = %0.2f' % (roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    #plt.title(plotTitle)
    plt.legend(loc="lower right", prop={'size': 12})
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.gca().legend(handles[::-1], labels[::-1])
    plt.tight_layout()
    plt.show(block=False)
    plt.savefig('figures/architectureComparison/suppl_'+figureFileName)

plt.show()
