#%%
import h5py, json
# import json
import csv
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
sys.path.insert(1, 'C:\\Users\\Znerp\\Documents\\GitHub\\kaggle-dsb2-keras\\')
from vis_utils import vis_learning_curve, vis_learning_curve_old, vis_learning_curve_seg



#%% 
base_dir = 'C:\\Users\\Znerp\\Documents\\GitHub\\kaggle-dsb2-keras\\training_results'

# #%% visualize learning curves
# dirnames = ['20181211_2037', '20181212_0920', '20181212_1228']

# for i,dirname in enumerate(dirnames):
#     filepath = os.path.join(base_dir, dirname, 'metadata.h5')

#     fig, (ax1, ax2) = vis_learning_curve(filepath)
    
#     # fig.savefig('..\\..\\figures\\test{}.png'.format(i))

#%% alternatively loop over all folders 
# nSkip = 7 # skip the first nSkip folders
# print('Skipping the first {} folders.'.format(nSkip))
nOld = 5
counter = 0
for _,dirnames,_ in os.walk(base_dir): # only need the directory names
    for dirname in dirnames:
        if dirname == 'segmentation':
            print('Skipping segmentation folder.')
            continue

        counter += 1

        if counter <= nOld: # old print version + metadata.h5
            metafile = os.path.join(base_dir, dirname, 'metadata.h5')
            try:
                fig, (ax1, ax2) = vis_learning_curve_old(metafile, fmt='h5')
            except Exception as e:
                print(dirname, e)
        elif counter > nOld and counter < 11: # old print version; json
            metafile = os.path.join(base_dir, dirname, 'metadata.json')
            # print(dirname)
            try:
                fig, (ax1, ax2) = vis_learning_curve_old(metafile)
            except Exception as e:
            # except FileNotFoundError as e:
                print(dirname, e)
        else: # new version
            metafile = os.path.join(base_dir, dirname, 'metadata.json')
            lossfile = os.path.join(base_dir, dirname, 'val_loss_all.csv')
            try:
                fig, (ax1, ax2) = vis_learning_curve(metafile, lossfile)
            except Exception as e:
                print(dirname, e)
        # fig.savefig('..\\..\\figures\\test{}.png'.format(i))


#%% visualization of segmentation training; all in one plot
seg_dir = os.path.join(base_dir, 'segmentation')

loss_best = []
Dirnames = []
for _,dirnames,_ in os.walk(seg_dir):
    for dirname in dirnames:
        # Dirnames.append(dirname)

        metafile = os.path.join(seg_dir, dirname, 'metadata.json')
        lossfile = os.path.join(seg_dir, dirname, 'val_loss_all.txt')

        # try:
        with open(metafile, 'r') as jf:
            metadict = json.load(jf)
            lr = metadict['hparms']['optimizer']['params']['lr']
            epochs = metadict['hparms']['nb_iter']
            try:
                loss_best = metadict['hparms']['loss_best']
            except KeyError:
                loss_best = None

        if loss_best != None:
            plt_tit = 'lr = {}, best loss = {:.4f}'.format(lr, loss_best)
        else:
            plt_tit = 'lr = {}, best loss = unknown'.format(lr)
        fig, ax = vis_learning_curve_seg(lossfile, plt_tit)
        # except Exception as e:
            # print(dirname, e)   


# for i in range(len(Dirnames)):
#     print(Dirnames[i], loss_best[i])


#%% debugging segmentation visualization
with open(lossfile, 'r') as csvf:
    csvr = csv.reader(csvf)
    loss = []
    for row in csvr:
        loss.append(row)
    loss = np.array([float(x[0]) for x in loss if len(x)!=0]) # filter empty lines
    print(loss)
        

# #%% what went wrong with a certain folder?
# f = h5py.File(os.path.join(base_dir, '20181213_0959', 'metadata.h5'), 'r')
# for bla in f.keys():
#     print(bla + ':\n')
#     for blubb in f[bla].keys():
#         print(blubb)
#     print('\n')

# #%% does plotting the title directly work?
# import numpy as np
# import matplotlib.pyplot as plt

# plt.plot(np.arange(1,101), np.linspace(2,200,100), title='blubb')

#%%
fig = plt.figure()
ax = plt.plot(np.arange(1,100+1), np.linspace(2,200,100))
print(fig)
print(type(ax))

#%%
