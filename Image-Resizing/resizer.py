
import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def show(data: list, masterTitle: str = "", titles: list = None,k=2):
    ''' Use matplotlib module to display images.
    '''
    fig = plt.figure()
    titles = titles or ["" for _ in range(len(data))]
    for i in range(len(data)):
        ax = fig.add_subplot(int(np.ceil(len(data)/k)),k,i+1)
        ax.set_title(titles[i])        
        ax.imshow(data[i])

    fig.suptitle(masterTitle)
    plt.show()

img = cv2.imread("drone/DJI_0406.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
I = img.copy()
data_cube,data_mitchellcubic,data_near = [],[],[]
pct = [.75,.5,.25,.1,.05,.01]
title = "Nearest"


for i in pct:
    for arr, m in zip([data_cube, data_mitchellcubic, data_near], 
        [tf.image.ResizeMethod.BICUBIC, tf.image.ResizeMethod.MITCHELLCUBIC,tf.image.ResizeMethod.NEAREST_NEIGHBOR]):

        arr.append(np.uint8(tf.image.resize(I, (int(I.shape[0]*i), int(I.shape[1]*i)), method=m)))

show(data_near,'',[f"{title} {i*100}%" for i in pct],k=3)
