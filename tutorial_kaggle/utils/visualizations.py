#%%
import h5py
import json
import matplotlib.pyplot as plt
import numpy as np
import os

#%%
def vis_learning_curve(filename, fmt='json', **kwargs):
    """
    Visualizes the learning curve: Needs access to losses.

    Parms:
        filename: path to file that contains the losses and hyperparameters of the learning process
        fmt: format of the saved file; currently supported: json and h5py
    """

    # read in file
    if fmt == 'h5':
        with h5py.File(filename, 'r') as h:
            loss_train = h['losses']['train'].value
            loss_val = h['losses']['val'].value
            try:
                lr = h['hparms']['optimizer']['params']['lr'].value
            except Exception: # old version
                lr = h['hparms']['lr_alpha'].value
            epochs = h['hparms']['nb_iter'].value
            batch_size = h['hparms']['batch_size'].value
    elif fmt == 'json':
        with open(filename, 'r') as jf:
            metadict = json.load(jf)
            loss_train = np.array(metadict['losses']['train'])
            loss_val = np.array(metadict['losses']['val'])
            lr = metadict['hparms']['optimizer']['params']['lr']
            epochs = metadict['hparms']['nb_iter']
            batch_size = metadict['hparms']['batch_size']
    else:
        raise Exception('Format {} is invalid.'.format(fmt))

    # visualize learning process
    fig , (ax1, ax2) = plt.subplots(nrows=1, ncols=2)

    fig.suptitle('lr = {}, batch_size = {}'.format(lr, batch_size))
    ax1.set_title('Systole')
    l1, = ax1.plot(np.arange(1,epochs+1), loss_train[:,0], 'b-', **kwargs)
    l2, = ax1.plot(np.arange(1,epochs+1), loss_val[:,0], 'r-', **kwargs)
    ax1.set_ylabel('Loss')
    ax1.set_xlabel('Epoch')
    ax1.legend([l1, l2], ['training', 'validation'])

    ax2.set_title('Diastole')
    l3, = ax2.plot(np.arange(1,epochs+1), loss_train[:,1], 'b-', **kwargs)
    l4, = ax2.plot(np.arange(1,epochs+1), loss_val[:,1], 'r-', **kwargs)
    ax2.set_xlabel('Epoch')
    ax2.legend([l3, l4], ['training', 'validation'])

    return fig, (ax1, ax2)


#%% 
base_dir = 'C:\\Users\\Znerp\\Documents\\GitHub\\kaggle-dsb2-keras\\training_results'

# #%% visualize learning curves
# dirnames = ['20181211_2037', '20181212_0920', '20181212_1228']

# for i,dirname in enumerate(dirnames):
#     filepath = os.path.join(base_dir, dirname, 'metadata.h5')

#     fig, (ax1, ax2) = vis_learning_curve(filepath)
    
#     # fig.savefig('..\\..\\figures\\test{}.png'.format(i))

#%% alternatively loop over all folders 
nSkip = 7 # skip the first nSkip folders
counter = 1
print('Skipping the first {} folders.'.format(nSkip))
for _,dirnames,_ in os.walk(base_dir): # only need the directory names
    for dirname in dirnames:
        if counter <= nSkip:
            counter += 1
            print('skip')
            continue
        filepath = os.path.join(base_dir, dirname, 'metadata.json')
        # print(dirname)
        try:
            fig, (ax1, ax2) = vis_learning_curve(filepath)
        # except Exception as e:
        except FileNotFoundError as e:
            print(dirname, e)
        
        # fig.savefig('..\\..\\figures\\test{}.png'.format(i))


#%% what went wrong with a certain folder?
f = h5py.File(os.path.join(base_dir, '20181213_0959', 'metadata.h5'), 'r')
for bla in f.keys():
    print(bla + ':\n')
    for blubb in f[bla].keys():
        print(blubb)
    print('\n')