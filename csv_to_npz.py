import os
import numpy as np
import pathfinder
from app import read_image

filenames =  os.listdir(pathfinder.DATA_PATH+'train-csv')
img_ids = [int(x.split('.')[0].split('-')[0]) for x in filenames]
img_ids = list(set(img_ids))

for i,id in enumerate(img_ids):
    try:
        im = read_image('train', id)
        np.savez(pathfinder.COMPRESSED_DATA_PATH + str(id) + '.npz', im.astype('float16'))
        print 1.*i/len(img_ids)
    except:
        print "PROBLEM"
