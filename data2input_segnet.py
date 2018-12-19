# 17.12.2018
# this script is supposed to convert the segmented images and their annotations 
# (courtesy of Julian de Wit) to a format the single-slice segnet can be trained with

import os
import numpy as np
import imageio
from scipy.ndimage import zoom

data_folder = 'C:\\Users\\Znerp\\Documents\\GitHub\\kaggle-dsb2-keras\\data\\segmentation\\images'
save_folder = 'C:\\Users\\Znerp\\Documents\\GitHub\\kaggle-dsb2-keras\\data\\segmentation'

img_shape_post = [184,184] # original shape 184x184


files = os.listdir(data_folder)

X_train = []
y_train = []
counter = 0
for filename in files:
    img = imageio.imread(os.path.join(data_folder, filename))
    
    # image size correction
    if img_shape_post != [184,184]:
        img = zoom(img, img_shape_post)

    if filename[-5] == 'o': # ground truth/annotated file
        img = img // 255
        y_train.append(np.array([img]))
        # print(filename, img.shape)
        # print(np.max(img))
    else: 
        X_train.append(np.array([img]))
    
    counter += 1
    if counter % 500 == 0:
        print('{} out of {} images processed.'.format(counter, len(files)))

print('Done processing. Saving images.')
np.save(os.path.join(save_folder, 'X_train_{}.npy'.format(img_shape_post[0])), X_train)
np.save(os.path.join(save_folder, 'y_train_{}.npy'.format(img_shape_post[0])), y_train)


### debugging:
# for image in y_train:
#     print(np.max(image))
#     # print(type(image))

# for i in range(lol.shape[0]):
#     print(np.max(lol[i,:,:,:]))