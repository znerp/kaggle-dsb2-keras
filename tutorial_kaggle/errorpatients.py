### 06.12.2018
# Script for getting more information about the patients that could not be properly read in by the DatasetSAX class
import os

data_dir = 'C:\\Users\\Znerp\\Documents\\GitHub\\kaggle-dsb2-keras\\data'

failedPatients = []

# # if file was actually saved with a delimiter between patients
# with open(os.path.join(data_dir,'failed.csv'), 'r') as csvfile:
#     content = csv.reader(csvfile)
#     for row in content:
#         failedPatients.append(row)

# for the current way where all patients and warnings are just one string
with open(os.path.join(data_dir,'failed.csv'), 'r') as csvfile:
    failed = csvfile.readlines()

failedPatients = failed[0].split(sep=")(" )
nFailedPatients = len(failedPatients)

for failedPatient in failedPatients:
    print(failedPatient)

# print(failedPatients)