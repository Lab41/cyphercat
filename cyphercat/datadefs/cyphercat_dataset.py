from cyphercat.definitions import DATASETS_DIR, DATASPLITS_DIR
from .cifar10_dataset import Cifar10_preload_and_split
from .libri_dataset import Libri_preload_and_split


class CCATDataset():
    """
    This is a generic cyphercat dataset class for accessing the various
    datasets accessible to the package.

    # Args
        name (string): dataset name
        splits (list): Dataframe with data path and metadata.
        transform List(torch): list of torch transform functions
                               Must be None or length must == {1, len(splits)}
        cache: bool. Whether or not to use the cached index file
    """
    def __init__(self, path='', name='', splits=[1], transforms=None):

        self.path = path
        self.name = name
        self.splits = splits
        self.nsplits = len(splits)
        self.transforms = transforms
        self.datasplits = self.prep_dataset_splits()
        
    def prep_dataset_splits(self):
        # Check that there is either 1 transform fn,
        # or the same as the number of requested splits
        if self.transforms:
            tlen = len(self.transforms)
            slen = self.nsplits
            assert tlen==1 or tlen == slen, "Error: transform list incorrect. "\
                                            "Must be 1 element or same length as splits. "\
                                            "len(transforms) == {}".format(tlen)

        # Grab appropriate preloader_splitter function
        presplit_fn = get_preload_split_fn(self.name)
        # Do the splits preloading...
        return presplit_fn(path=self.path, splits=self.splits, transform=self.transforms)

    def get_dataset_all_splits(self):
        return self.datasplits

    def get_split_n(self, n=0):
        assert n >= 0 and n < self.nsplits, "Error: requesting invalid split."\
                                           "Choose split btw 0 and {}".format(self.nsplits-1)
        return self.datasplits[n]


# Functions 
PRELOAD_SPLIT_FN_DICT = {'cifar-10': Cifar10_preload_and_split,
                         'librispeech': Libri_preload_and_split,
                         }


def get_preload_split_fn(name=''):
    """
    Convenience function for retrieving allowed
    cyphercat split dataset functions.

    Parameters
    ----------
    name : {'cifar-10', 'librispeech'}
          Name of dataset

    Returns
    -------
    fn : function
         Dataset specific splitter function
    """
    if name in PRELOAD_SPLIT_FN_DICT:
        fn = PRELOAD_SPLIT_FN_DICT[name]
        return fn
    else:
        raise ValueError('Invalid dataset, {}, entered. Must be '
                         'in {}'.format(name, PRELOAD_SPLIT_FN_DICT.keys()))
