'''
image_helper.py : contains loadgray() method to load an RGBA image in PNG format 
and return a grayscale image

Copyright (C) 2017 Jack Baird, Alex Cantrell, Keith Denning, Rajwol Joshi, 
Simon D. Levy, Will McMurtry, Jacob Rosen

This file is part of AirSimTensorFlow

MIT License
'''

import matplotlib.pyplot as plt

# Images are too big to train quickly, so we scale 'em down
SCALEDOWN = 6

# Where we'll store images
IMAGEDIR = './carpix'

# Create images directory if it doesn't exist
def loadgray(filename):
    '''
    Loads an RGBA image from FILENAME, converts it to grayscale, and returns a flattened copy
    '''
    
    image = plt.imread(filename)

    # RGB -> gray formula from https://www.johndcook.com/blog/2009/08/24/algorithms-convert-color-grayscale/
    image = 0.21 * image[:,:,0] + 0.72 * image[:,:,1] + 0.07 * image[:,:,2]
    image = image[0::SCALEDOWN, 0::SCALEDOWN]
    image = image.flatten()

    return image
