# Import all necessary packages
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


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

def remove_dublicates(List): # Function for removing dublicates
  return list(dict.fromkeys(List))

## 
def check_neighbours(img,list_ij):
    max_size = np.shape(img)
    list_new = []
    for index in list_ij:
        coor = [(index[0],index[1]+1),(index[0],index[1]-1),(index[0]+1,index[1]),(index[0]-1,index[1])]
        val = []
        #Remove edge coordinates
        if index[0] == 0:
            coor.pop(3)
        elif index[0] == max_size[0]:
            coor.pop(2)
        if index[1] == 0:
            coor.pop(1)
        elif index[1] == max_size[1]:
            coor.pop(0)    
        ##
        for i in range(len(coor)):
            try:
                val.append(img[coor[i]])
            except:
                val.append(0)
        for i in range(len(coor)):
            list_new.append(coor[i]*int(val[i]))
    list_new = remove_dublicates(list_new) # Remove dublicates
    return [i for i in list_new if i != ()] # Remove empty bracets from multiplication with 0

def count_elements(img): # Function for counting elements on image
    elements = 0
    num_of_elements = 0
    img_original = img
    img_count = np.zeros(np.shape(img))
    for i in range(np.shape(img)[0]):
        for j in range(np.shape(img)[1]):
            if img[i,j] == 1:
                object = True
                list_ij = [(i,j)]
                while object == True:
                    for index in list_ij:
                        img_count[index] = 1
                    img = img_original-img_count
                    list_ij = check_neighbours(img,list_ij)
                    if list_ij == []:
                        object = False
                elements += img_count
                num_of_elements += 1
    return num_of_elements, elements

def apply_colors(img):
    colors = ['black', 'red', 'blue', 'green', 'yellow', 'orange','purple']
    num_colors = len(colors)
    
    # Create a custom colormap
    cmap = ListedColormap(colors)
    
    # Plot the image
    plt.imshow(np.where(img== 0, 0, img % (num_colors - 1) + 1), cmap=cmap, interpolation='nearest')
    plt.show()