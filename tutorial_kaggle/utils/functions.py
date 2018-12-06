import os
from scipy.ndimage import zoom
from warnings import warn
from classes import DatasetSAX


def rezoom(in_data, t_dim=30, x_dim=128, y_dim=128):
    return zoom(in_data.images, [1,
                                    t_dim/in_data.images.shape[1], 
                                    x_dim/in_data.images.shape[2], 
                                    y_dim/in_data.images.shape[3]])


def read_and_process(in_path, t_dim=30, x_dim=128, y_dim=128, withErrCatch=False): # reads data in with pre-defined function and scales it to 128x128
    """Function that creates an object of the DatasetSAX class and processes it further.
    Zooms every image to the size t_dim*x_dim*y_dim.
    
    Input:
        in_path:        Path to the patient (parent) folder.
        t_dim:          Number of time frames of the output image sequence.
        x_dim:          Width of the output images.
        y_dim:          Height of the output images.
        withErrCatch:   Specifies if errors should be caught (True) or stop code execution (False)

    Output:
        list of in_path, zoom_time, area_multiplier and zoomed image stack. 

        zoom_time is a list of the new time stamps of the slices (if != t_dim; shrinks/expands it to t_dim) 
        area_multiplier is the area of the image in mm (calculated from metadata PixelSpacing)
    """
    if withErrCatch:
        try:
            cur_data = DatasetSAX(in_path, os.path.basename(in_path)) 
            # os.path.basename gets the name of the lowest folder
            cur_data.load()
            if cur_data.time is not None: # when would that ever be none? only if no sax data was found? but then there would be a problem anyway!?
                zoom_time = zoom(cur_data.time, [t_dim/len(cur_data.time)]) 
            else:
                zoom_time = range(t_dim)
            return [in_path, zoom_time, cur_data.area_multiplier, rezoom(cur_data, t_dim=30, x_dim=128, y_dim=128)] # scale single images to size (?,30,128,128)
            #return {'path': in_path, 'time': zoom_time, 'area': cur_data.area_multiplier, 'images': rezoom(cur_data, t_dim=30, x_dim=128, y_dim=128)} 
        except Exception as e: # catches exceptions without letting them stop the code
            warn('{}'.format(e), RuntimeWarning)
            return None
    else:
        cur_data = DatasetSAX(in_path, os.path.basename(in_path)) 
        cur_data.load()
        if cur_data.time is not None: # when would that ever be none? only if no sax data was found? but then there would be a problem anyway!?
            zoom_time = zoom(cur_data.time, [t_dim/len(cur_data.time)]) 
        else:
            zoom_time = range(t_dim)
        return [in_path, zoom_time, cur_data.area_multiplier, rezoom(cur_data, t_dim, x_dim, y_dim)] # scale images
        #return {'path': in_path, 'time': zoom_time, 'area': cur_data.area_multiplier, 'images': rezoom(cur_data, t_dim=30, x_dim=128, y_dim=128)} 