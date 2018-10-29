from __future__ import print_function

import os
import sys
import yaml


# Ensure basic, necessary fields are in the config file
def check_fields(cfg=None, tset=None):
    seen = set()
    for key, value in cfg.items():
        seen.add(key)

    return tset.issubset(seen)


class Configurator(object):
    """
    Configuration file reader
    """

    # Fields, subfields required in configuration file
    reqs = set(["data"])

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
        self.dataset = cfg['data']
        #self.avail_models = cfg.get('models_to_run', '').split(',')
        #self.head_outpath = cfg.get('outpath', os.path.join(self.datapath, 'saved_models'))



class DataStruct(object):
    """
    Expected directory structure
    for accessing image data sets
    """
    
    reqs = set(["name", "datapath", "classes", "height", "width", "channels"])
    
    def __init__(self, dataset=None):

        self.dataset = dataset
        
        if not check_fields(cfg=dataset, tset=self.reqs):
            raise AssertionError("Some subfields under 'data' field not found. "
                                 "Required fields: {}".format(self.reqs))

        self.name      = dataset.get('name')
        self.data_path = dataset.get('datapath')
        self.url       = dataset.get('url')
        self.height    = int(dataset.get('height'))
        self.width     = int(dataset.get('width'))
        self.channels  = int(dataset.get('channels'))
        self.classes   = int(dataset.get('classes'))
        self.labels    = dataset.get('labels', self.default_labels()).replace(" ", "").split(',')

        #self.train_dir = os.path.join(head_dir, 'training')
        #self.test_dir  = os.path.join(head_dir, 'testing')
        #self.check_dirs()

    def default_labels(self):
        return str(list(range(0, self.classes))).strip('[]')

    def check_dirs(self):
        if not os.path.exists(self.head_dir):
            print('No such directory {} '
            'does not exist!'.format(self.head_dir), file=sys.stderr)
            sys.exit()

        if not os.path.exists(self.train_dir):
            print('No such directory {} '
            'does not exist!'.format(self.train_dir), file=sys.stderr)
            sys.exit()

        if not os.path.exists(self.test_dir):
            print('No such directory {} '
            'does not exist!'.format(self.test_dir), file=sys.stderr)
            sys.exit()
