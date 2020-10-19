# 4-1_triageModelCalibration.py
#
# This script calibrates the semi automated models
#

import pandas as pd
import pickle
import numpy as np

calibrationCohort = pickle.load( open( "data/slideLevelAggregation/calibrationCohort.p", "rb" ) )

qcScore = pd.read_csv('data/slideLevelAggregation/HE-data-vgg16.csv')
qcScore['Case'] = qcScore['Case'].str.replace('_HE_1', '')
qcScoreColumn = 'Gastric count (> 0.99)'

diagScore = pd.read_csv('data/slideLevelAggregation/TFF3-data-resnet18.csv')
diagScore['Case'] = diagScore['Case'].str.replace('_TFF3_1', '')
diagScoreColumn = 'TFF3 positive count (> 0.93)'

mergedScores = pd.merge(qcScore[['Case',qcScoreColumn,'Cytosponge QC']],diagScore[['Case',diagScoreColumn,'Endoscopy (at least C1M3)','Endoscopy (at least C1M1)','Endoscopy (at least C1)','Endoscopy (at least C2)','Endoscopy (at least C3)','Cytosponge','Endoscopy (at least C1 or M3) + Biopsy (IM)']],on="Case")

mergedScores['DL-QC-L'] = (mergedScores[qcScoreColumn]>=1).astype(int) # Consensus of all four observers
mergedScores['DL-QC-H'] = (mergedScores[qcScoreColumn]>=49).astype(int) # Consensus of all four observers
mergedScores['DL-Diag-L'] = (mergedScores[diagScoreColumn]>=1).astype(int) # Consensus of all four observers
mergedScores['DL-Diag-H'] = (mergedScores[diagScoreColumn]>=11).astype(int) # Consensus of all four observers


mergedScores['Triage class'] = 0
# Kick out no confidence
for i, row in mergedScores.iterrows():
    if row['DL-QC-L'] == 0 and row['DL-Diag-L'] == 0:
        mergedScores.at[i,'Triage class'] = 0
    elif row['DL-QC-H'] == 1 and row['DL-Diag-L'] == 0:
        mergedScores.at[i,'Triage class'] = 1
    elif row['DL-QC-L'] == 1 and row['DL-QC-H'] == 0 and row['DL-Diag-L'] == 0:
        mergedScores.at[i,'Triage class'] = 2
    elif row['DL-QC-L'] == 0 and row['DL-Diag-L'] == 1:
        mergedScores.at[i,'Triage class'] = 3
    elif row['DL-QC-L'] == 1 and row['DL-QC-H'] == 0 and row['DL-Diag-L'] == 1 and row['DL-Diag-H'] == 0:
        mergedScores.at[i,'Triage class'] = 4
    elif row['DL-QC-H'] == 1 and row['DL-Diag-L'] == 1 and row['DL-Diag-H'] == 0:
        mergedScores.at[i,'Triage class'] = 5
    elif row['DL-QC-L'] == 1 and row['DL-QC-H'] == 0 and row['DL-Diag-H'] == 1:
        mergedScores.at[i,'Triage class'] = 6
    elif row['DL-QC-H'] == 1 and row['DL-Diag-H'] == 1:
        mergedScores.at[i,'Triage class'] = 7

mergedScoresCalibration = mergedScores[mergedScores['Case'].isin(calibrationCohort)]
mergedScoresValidation = mergedScores[~mergedScores['Case'].isin(calibrationCohort)]

mergedScoresCalibration.to_excel("data/triageOutput/calibrationOutput.xlsx")
mergedScoresValidation.to_excel("data/triageOutput/validationOutput.xlsx")

print(len(mergedScoresValidation.loc[mergedScoresValidation['Endoscopy (at least C1 or M3) + Biopsy (IM)']==True]))
print(len(mergedScoresValidation.loc[mergedScoresValidation['Endoscopy (at least C1 or M3) + Biopsy (IM)']==False]))
print(len(mergedScoresCalibration.loc[mergedScoresCalibration['Endoscopy (at least C1 or M3) + Biopsy (IM)']==True]))
print(len(mergedScoresCalibration.loc[mergedScoresCalibration['Endoscopy (at least C1 or M3) + Biopsy (IM)']==False]))

print(len(mergedScoresCalibration))
