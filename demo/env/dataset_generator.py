
import random

WIDTH = 10
HEIGHT = 10

data_set = []

for x in range(WIDTH):
    for y in range(HEIGHT):
        # ndvi is known to be indirectly proportional with lst
        ndvi = random.uniform(0,1)
        lst = random.uniform(0, 10)
        actual_fire = True if ndvi > 0.4 else False
        data_set.append([x,y, ndvi, lst, actual_fire])

