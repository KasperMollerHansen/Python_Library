# Import all necessary packages
import warnings
import numpy as np


################################
def rgb2gray(rgb, coefficient = [0.2989, 0.5870 , 0.1141]):
    if np.round(sum(coefficient),4) != 1:
        warnings.warn("Coefficients doesn't add up to 1. Results may not be accurate.", UserWarning)
    
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = coefficient[0] * r + coefficient[1] * g + coefficient[2] * b

    return gray

def background_equalizer(img,x_thresh = 0.4,y_thresh = 0.4, thresh = 15): # Function for equalizing background in gray scale image
    background = np.zeros(np.shape(img))
    n,m = np.shape(img)
    for i in range(n):
        quan_x = np.quantile((img[i,:]),x_thresh)
        background[i,:] += quan_x
    for i in range(m):
        quan_y = np.quantile((img[:,i]),y_thresh)
        background[:,i] += quan_y
    background /=2 # Every point have 2 contributions
    img_new = img-background
    img_new[img_new<0]=0
    img_bw = img_new > thresh
    return background,img_new, img_bw