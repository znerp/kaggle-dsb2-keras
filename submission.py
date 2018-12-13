import csv
import numpy as np
import os
import h5py

import sys
sys.path.insert(1, 'C:\\Users\\Znerp\\Documents\\GitHub\\kaggle-dsb2-keras\\model')
from model import get_model
from git_utils import real_to_cdf, preprocess


def load_validation_data(filename):
    """
    Loads validation data from .h5py files.
    """
    with h5py.File(filename, 'r') as w:
        X = w['image'].value
        ids = w['id'].value
        area_mult = w['area_multiplier'].value

    a_correct = np.reshape(area_mult, (area_mult.shape[0], 1))

    X = X.astype(np.float32)
    X /= 255

    return X, ids, a_correct


def accumulate_study_results(ids, prob):
    """
    Accumulate results per study (because one study has many SAX slices),
    so the averaged CDF for all slices is returned.
    """
    sum_result = {}
    cnt_result = {}
    size = prob.shape[0]
    for i in range(size):
        study_id = ids[i]
        idx = int(study_id)
        if idx not in cnt_result:
            cnt_result[idx] = 0.
            sum_result[idx] = np.zeros((1, prob.shape[1]), dtype=np.float32)
        cnt_result[idx] += 1
        sum_result[idx] += prob[i, :]
    for i in cnt_result.keys():
        sum_result[i][:] /= cnt_result[i]
    return sum_result

# metavariables
N_PATIENTS = 200
weight_folder = 'C:\\Users\\Znerp\\Documents\\GitHub\\kaggle-dsb2-keras\\model'
data_folder = 'C:\\Users\\Znerp\\Documents\\GitHub\\kaggle-dsb2-keras\\data'
val_folder = os.path.join(data_folder, 'validate') 
submission_folder = os.path.join(data_folder, 'submission') 
val_file = os.path.join(val_folder, 'validate_mri_64_64_N{}.h5'.format(N_PATIENTS))

print('Starting prediction and submission for ' + str(N_PATIENTS) + ' Patients.')

subm_file = os.path.join(submission_folder, 'submission.csv')
if os.path.isfile(subm_file):
    overwrite = input('Do you really want to overwrite {}? [y/n]'.format(subm_file))
    if overwrite != 'y':
        raise RuntimeError('Code stopped. File will not be overwritten.')

print('Loading and compiling models...')
model_systole = get_model()
model_diastole = get_model()

print('Loading models weights...')
model_systole.load_weights(os.path.join(weight_folder, 'weights_systole_best.hdf5'))
model_diastole.load_weights(os.path.join(weight_folder, 'weights_diastole_best.hdf5'))

# load val losses to use as sigmas for CDF
with open(os.path.join(weight_folder, 'val_loss.txt'), mode='r') as f:
    val_loss_systole = float(f.readline())
    val_loss_diastole = float(f.readline())

print('Loading validation data...')
X, ids, a_correct = load_validation_data(val_file) 

print('Pre-processing images...')
X = preprocess(X)

batch_size = 32
print('Predicting on validation data...')
pred_systole = model_systole.predict(X, batch_size=batch_size, verbose=1)
pred_diastole = model_diastole.predict(X, batch_size=batch_size, verbose=1)
# correction for area multiplier required 
pred_systole = pred_systole * a_correct
pred_diastole = pred_diastole * a_correct

# real predictions to CDF
cdf_pred_systole = real_to_cdf(y=pred_systole, sigma=val_loss_systole)
cdf_pred_diastole = real_to_cdf(pred_diastole, val_loss_diastole)

print('Accumulating results...')
sub_systole = accumulate_study_results(ids, cdf_pred_systole)
sub_diastole = accumulate_study_results(ids, cdf_pred_diastole)

# write to submission file
print('Writing submission to file...')
fi = csv.reader(open(os.path.join(submission_folder, 'sample_submission_validate.csv')))
f = open(os.path.join(submission_folder, 'submission.csv'), 'w')
fo = csv.writer(f, lineterminator='\n')
# next(fi) # skip first line as it is basically a header

for it,line in enumerate(fi):
    if it == 0:
        fo.writerow(line)
    else:
        idx = line[0]
        key, target = idx.split('_')
        key = int(key)
        out = [idx]
        if key in sub_systole:
            if target == 'Diastole':
                out.extend(list(sub_diastole[key][0])) # like append, but extends a list by another list instead of a single element(which could also be a list --> list within a list)
            else:
                out.extend(list(sub_systole[key][0]))
        else:
            print('Miss {0}'.format(idx))
        fo.writerow(out)
f.close()

print('Done.')
