import numpy as np

NORTH = 0
SOUTH = 1
EAST = 2
WEST =3

MODEL_PATH = "learned_model"

def argmax(numpy_array):
    """ argmax implementation that chooses randomly between ties """
    maxes = np.argwhere(numpy_array == np.amax(numpy_array))
    maxlist = maxes.flatten().tolist()
    val = np.random.choice(maxlist)
    return val