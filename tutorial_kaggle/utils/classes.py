import pydicom as dicom
import re
import os
import numpy as np

# only deals with the SAX images!
class DatasetSAX(object): 
    """
    Class for creating datasets from the SAX images.

    Attributes:
        directory
        time
        slices
        slices_map
        name
        images
        dist
        area_multiplier
        metadata
        
    Methods:
        __init__
        _filename
        _read_dicom_image
        _read_all_dicom_images
        load
    """

    def __init__(self, directory, subdir): # subdir = patient, directory = directory of patient data
        """Initiator: Initializes the dataset object with the path to the patient data. 
        Furthermore extracts time and SAX slices."""

        # deal with any intervening directories
        while True:
            subdirs = next(os.walk(directory))[1] # return dir name(s)
            if len(subdirs) == 1: # if only one folder is contained --> go deeper
                directory = os.path.join(directory, subdirs[0])
            else:
                break

        slices = []
        for s in subdirs: # gets the SAX slices of the patient
            m = re.match("sax_(\d+)", s)
            if m is not None:
                slices.append(int(m.group(1)))

        slices_map = {}
        first = True
        times = []
        for s in slices:
            files = next(os.walk(os.path.join(directory, "sax_%d" % s)))[2] # gets filenames in sax folder; os.walk -> (dirpath, dirnames, filenames)
            offset = None

            for f in files: # loops over the SAXslices 
                m = re.match("IM-(\d{4,})-(\d{4})\.dcm", f) # brackets define groups that can be accessed after matching
                if m is not None:
                    if first:
                        times.append(int(m.group(2))) # part after the second dash
                    if offset is None: 
                        offset = int(m.group(1)) # part after the first dash; apparently indicates offset

            first = False
            slices_map[s] = offset # save offsets to the map of slices

        self.directory = directory
        self.name = subdir
        # print(self.directory, self.name)
        self.folders = os.listdir(self.directory)
        # print(self.folders)
        self.time = sorted(times)
        self.slices = sorted(slices)
        self.slices_map = slices_map

    def _filename(self, s, t): # s = slice, t = time
        return os.path.join(self.directory,"sax_%d" % s, "IM-%04d-%04d.dcm" % (self.slices_map[s], t))

    def _read_dicom_image(self, filename):
        d = dicom.read_file(filename)
        img = d.pixel_array # pixel_array contains the pixels, ergo the image 
        return np.array(img)

    def _read_all_dicom_images(self):
        f1 = self._filename(self.slices[0], self.time[0])
        d1 = dicom.read_file(f1)
        (x, y) = d1.PixelSpacing
        (x, y) = (float(x), float(y)) # conversion from string
        f2 = self._filename(self.slices[1], self.time[0])
        d2 = dicom.read_file(f2)

        # try a couple of things to measure distance between slices
        try:
            dist = np.abs(d2.SliceLocation - d1.SliceLocation)
        except AttributeError:
            try:
                dist = d1.SliceThickness
            except AttributeError:
                dist = 8  # better than nothing...

        self.images = np.array([[self._read_dicom_image(self._filename(d, i))
                                 for i in self.time]
                                for d in self.slices]) # all time steps should have the same slices, right?
        self.dist = dist
        self.area_multiplier = x * y
        self.metadata = {"PatientAge": d1.PatientAge, "PatientBirthDate": d1.PatientBirthDate, "PatientName": d1.PatientName, 
                         "PatientSex": d1.PatientSex, "PatientPosition": d1.PatientPosition,}

    def load(self):
        self._read_all_dicom_images()
