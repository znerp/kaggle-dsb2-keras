# for checking why so many patients are missing in the submission file
import h5py

filename = 'data\\validate\\validate_mri_64_64_N200.h5'
with h5py.File(filename, 'r') as w:
    X = w['image'].value
    ids = w['id'].value
    area_mult = w['area_multiplier'].value

unique_ids = list(set(ids))
unique_ids.sort()
print(unique_ids)
print(len(unique_ids))

print(ids)