import matplotlib.pyplot as plt

# Images are too big to train quickly, so we scale 'em down
SCALEDOWN = 6

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
