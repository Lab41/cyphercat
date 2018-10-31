from __future__ import print_function

import io
import os


# Dictionary printer
def print_dict(dct):
    for key, value in sorted(dct.items(), reverse=True):
        print("{}: {}".format(key, value))

# Set string printer
def set_to_string(iset=None):
    sstr = ', '.join([str(i) for i in iset])
    return sstr

# Dictionary string key-printer
def keys_to_string(struct=None):
    kstr = ', '.join([k for k in struct.keys()])
    return kstr

# Color mode dictionary for specifying
# color_mode in data generators
color_mode_dict = {1 : 'grayscale',
                   3 : 'rgb'}

