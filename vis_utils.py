# 18.12.2018 
# Some tools for visualization of the learning process

import h5py, json
import csv
import matplotlib.pyplot as plt
import numpy as np
import os


def vis_learning_curve(metafile, lossfile, fmt='json', **kwargs):
    """
    Visualizes the learning curve: Needs access to losses.

    Parms:
        metafile: path to file that contains the losses and hyperparameters of the learning process
        lossfile: csvfile containing the loss values
        fmt: format of the saved file; currently supported: json and h5py
    """

    # read in file
    if fmt == 'h5':
        with h5py.File(metafile, 'r') as h:
            try:
                lr = h['hparms']['optimizer']['params']['lr'].value
            except Exception: # old version
                lr = h['hparms']['lr_alpha'].value
            epochs = h['hparms']['nb_iter'].value
            batch_size = h['hparms']['batch_size'].value
    elif fmt == 'json':
        with open(metafile, 'r') as jf:
            metadict = json.load(jf)
            lr = metadict['hparms']['optimizer']['params']['lr']
            epochs = metadict['hparms']['nb_iter']
            batch_size = metadict['hparms']['batch_size']
    else:
        raise Exception('Format {} is invalid.'.format(fmt))

    with open(lossfile, 'r') as csvf:
        csvr = csv.reader(csvf)
        loss = []
        for row in csvr:
            loss.append(row)
        loss=[x for x in loss if len(x)!=0] # filter empty lines
        loss_train = np.array([ [float(x[0]), float(x[1])] for x in loss])
        loss_val = np.array([ [float(x[2]), float(x[3])] for x in loss])

    # visualize learning process
    fig , (ax1, ax2) = plt.subplots(nrows=1, ncols=2)

    fig.suptitle('lr = {}, batch_size = {}'.format(lr, batch_size))
    ax1.set_title('Systole')
    l1, = ax1.plot(np.arange(1,len(loss_train)+1), loss_train[:,0], 'b-', **kwargs)
    l2, = ax1.plot(np.arange(1,len(loss_val)+1), loss_val[:,0], 'r-', **kwargs)
    ax1.set_ylabel('Loss')
    ax1.set_xlabel('Epoch')
    ax1.legend([l1, l2], ['training', 'validation'])

    ax2.set_title('Diastole')
    l3, = ax2.plot(np.arange(1,len(loss_train)+1), loss_train[:,1], 'b-', **kwargs)
    l4, = ax2.plot(np.arange(1,len(loss_val)+1), loss_val[:,1], 'r-', **kwargs)
    ax2.set_xlabel('Epoch')
    ax2.legend([l3, l4], ['training', 'validation'])

    return fig, (ax1, ax2)



def vis_learning_curve_old(metafile, fmt='json', **kwargs):
    """
    Visualizes the learning curve: Needs access to losses.
    Old version (before evening of 13.12.12).

    Parms:
        metafile: path to file that contains the losses and hyperparameters of the learning process
        fmt: format of the saved file; currently supported: json and h5py
    """

    # read in file
    if fmt == 'h5':
        with h5py.File(metafile, 'r') as h:
            loss_train = h['losses']['train'].value
            loss_val = h['losses']['val'].value
            try:
                lr = h['hparms']['optimizer']['params']['lr'].value
            except Exception: # old version
                lr = h['hparms']['lr_alpha'].value
            epochs = h['hparms']['nb_iter'].value
            batch_size = h['hparms']['batch_size'].value
    elif fmt == 'json':
        with open(metafile, 'r') as jf:
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



def vis_learning_curve_seg(lossfile, plt_tit, **kwargs):
    """
    vis_learning_curve_seg visualizes the learning curve of a segmentation network.

    Parms:
        lossfile: csvfile containing the loss values
        plt_tit: title of the plot
    """

    # read in loss file
    with open(lossfile, 'r') as csvf:
        csvr = csv.reader(csvf)
        loss = []
        for row in csvr:
            loss.append(row)
        loss = np.array([float(x[0]) for x in loss if len(x)!=0]) # filter empty lines
        

    # visualize learning process
    fig, ax = plt.subplots()
    ax = plt.plot(np.arange(1,len(loss)+1), loss, **kwargs)

    plt.title(plt_tit)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.show()

    return fig, ax

