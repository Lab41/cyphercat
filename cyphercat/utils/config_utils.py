from __future__ import print_function

import os
import sys
import yaml
from .utils import set_to_string, keys_to_string, color_mode_dict
from cyphercat.definitions import REPO_DIR



# Ensure basic, necessary fields are in the config file
def check_fields(cfg=None, tset=None):
    seen = set()
    for key, value in cfg.items():
        seen.add(key)

    return tset.issubset(seen)

# Test if path is absolute or relative
def test_abs_path(self, path=''):
    if path.startswith('/'):
        return path
    else:
        return os.path.join(REPO_DIR, path)

class Configurator(object):
    """
    Configuration file reader
    """

    # Fields, subfields required in configuration file
    reqs = set(["data", "train"])

    def __init__(self, config_file=""):

        # Get configuration file
        self.filepath = os.path.abspath(config_file)
        with open(config_file, 'r') as ymlfile:
            cfg = yaml.load(ymlfile)

        # Loaded config object
        self.cfg = cfg

        # Ensure necessary header fields exist
        if not check_fields(cfg=cfg, tset=self.reqs):
            raise AssertionError("Some fields in {} not found. "
                                 "Required fields: {}".format(self.filepath,
                                                              self.reqs))

        # Extract config parameters
        self.dataset       = cfg['data']
        self.train_model   = cfg['train']
        #self.avail_models = cfg.get('models_to_run', '').split(',')
        #self.head_outpath = cfg.get('outpath', os.path.join(self.datapath, 'saved_models'))



class ModelConfig(object):
    """
    Expected information defining a model.
    """
    reqs = set(["model", "runtrain"])

    def __init__(self, modelconfig=None):

        self.modelconfig = modelconfig
        
        if not check_fields(cfg=modelconfig, tset=self.reqs):
            raise AssertionError("Some subfields for 'model' field not found.\n"
                                 "  Required fields: {}\nExiting...\n".format(set_to_string(self.reqs)))
        self.name       = modelconfig.get('model')
        self.runtrain   = modelconfig.get('runtrain')
        self.model_path = test_abs_path(modelconfig.get('modelpath'))
        self.epochs     = modelconfig.get('epochs')
        self.batchsize  = modelconfig.get('batchsize')
        self.learnrate  = modelconfig.get('learnrate')



class DataStruct(object):
    """
    Expected directory structure
    for accessing image data sets.
    Generalization to data & audio forthcoming.
    """
   
    # Mandatory fields for 'data' yaml config file keyword
    reqs = set(["name", "datapath", "datatype"])
    image_reqs = set(["nclasses", "height", "width", "channels"])
    audio_reqs = set(["length", "seconds"])
    data_reqs = [image_reqs, audio_reqs]

    # Data types dictionary
    data_type_dict = {"image" : 0,
                      "audio" : 1}

    
    def __init__(self, dataset=None):

        self.dataset = dataset
        
        if not check_fields(cfg=dataset, tset=self.reqs):
            raise AssertionError("Some subfields under 'data' field not found.\n"
                                 "  Required fields: {}\nExiting...\n".format(set_to_string(self.reqs)))

        self.name      = dataset.get('name')
        self.data_path = test_abs_path(dataset.get('datapath'))
        self.data_type = dataset.get('datatype').lower()
        self.url       = dataset.get('url', '')
        self.save_path = os.path.join(self.data_path, self.name)
      
        # Ensure data type is permitted
        if (self.data_type not in self.data_type_dict):
            print("\nUnknown data type '{}'!\n  Allowed data types: {}\nExiting...\n".format(self.data_type,
                                                                                             keys_to_string(self.data_type_dict)))
            sys.exit()

        # Get index from dictionary to access specific data reqs
        dtype_ind = self.data_type_dict[self.data_type]

        # Check subfields
        if not check_fields(cfg=dataset, tset=self.data_reqs[dtype_ind]):
            raise AssertionError("\nSome subfields under 'data' field not found.\n "
                                 "  Required fields for {} data: {}\nExiting...\n".format(self.data_type,
                                                                                          set_to_string(self.data_reqs[dtype_ind])))

        # Image data specific 
        if dtype_ind == 0: 
            self.height     = int(dataset.get('height'))
            self.width      = int(dataset.get('width'))
            self.channels   = int(dataset.get('channels'))
            self.n_classes    = int(dataset.get('nclasses'))
            self.color_mode = color_mode_dict[self.channels]
            self.labels     = dataset.get('labels', self.default_labels()).replace(" ", "").split(',')

        # Audio data specific 
        elif dtype_ind == 1:
            self.length  = float(dataset.get('length'))
            self.seconds = float(dataset.get('seconds'))
    
    # Consecutive integers default data labels 
    def default_labels(self):
        return str(list(range(0, self.n_classes))).strip('[]')

