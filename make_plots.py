import numpy as np 

import app
import utils_plots




def _plt_all_ims(): 
    img_ids = app.temporary_get_img_ids()

    ch0_ims = []
    ch1_ims = []
    for img_id in img_ids:
    	print img_id
        img = app.read_image('train', img_id)
        ch0_ims.append(img[0].flatten())
        ch1_ims.append(img[1].flatten())
        utils_plots.plot_img(img,'plots/'+str(img_id)+'.jpg')

    tensor = np.concatenate(ch0_ims)
    for p in range(5,100,5):
    	print p, np.percentile(tensor,p)
    
    tensor = np.concatenate(ch1_ims)
    for p in range(5,100,5):
    	print p, np.percentile(tensor,p)



_plt_all_ims()