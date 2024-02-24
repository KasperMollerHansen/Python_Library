# Import all necessary packages
import warnings
import cv2
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

def background_equalizer(img, x_thresh=0.4, y_thresh=0.4, scale = 3, thresh=190):
    # Calculate row-wise and column-wise quantiles
    row_quantiles = np.quantile(img, x_thresh, axis=1)
    col_quantiles = np.quantile(img, y_thresh, axis=0)
    # Calculate mean background
    background = (row_quantiles[:, np.newaxis] + col_quantiles) / 2.0
    # Compute the difference between the image and the background
    img_new = (img - background)*scale
    # Set negative values resulting from subtraction to 0
    img_new = np.clip(img_new, 0, 255)
    # Binarize the resulting image based on the provided threshold
    img_bw = img_new > thresh
    return background, img_new, img_bw

def count_elements(img): # Function for counting elements on image
    def check_neighbours(img, list_ij):
        max_size = img.shape
        list_new = set()  # Using set for faster duplicate removal
        
        for index in list_ij:
            x, y = index
            neighbors = [(x, y+1), (x, y-1), (x+1, y), (x-1, y)]
            
            for coor in neighbors:
                i, j = coor
                if 0 <= i < max_size[0] and 0 <= j < max_size[1]:
                    val = img[i, j]
                    if val != 0:  # Avoid multiplication by zero
                        list_new.add(coor * int(val))
        return list(list_new)
    elements = 0
    num_of_elements = 0
    img_original = img
    img_count = np.zeros(np.shape(img))
    for i in range(np.shape(img)[0]):
        for j in range(np.shape(img)[1]):
            if img[i,j] == 1:
                obj = True
                list_ij = [(i,j)]
                while obj == True:
                    for index in list_ij:
                        img_count[index] = 1
                    img = img_original-img_count
                    list_ij = check_neighbours(img,list_ij)
                    if list_ij == []:
                        obj = False
                elements += img_count
                num_of_elements += 1
    return num_of_elements, elements

def apply_colors(img):
    colors = ['black', 'red', 'blue', 'purple','darkgreen', 'yellow', 'orange','cyan','magenta','lime']
    num_colors = len(colors)
    cmap = ListedColormap(colors)
    # Plot the image
    plt.imshow(np.where(img== 0, 0, img % (num_colors - 1) + 1), cmap=cmap, interpolation='nearest')
    plt.show()

def remove_text(img,kernel,window_dim,val_thresh=255):
    
    def erode_dilate_img(window,kernel,val_thresh):
        num = num_above_thresh(window, val_thresh)
        if num >= 1:
            window = cv2.erode(window*1.00, kernel, iterations=3)
            window = cv2.dilate(window*1.00, kernel, iterations=1)
        return window

    def num_above_thresh(matrix, threshold):
        count = 0
        for row in matrix:
            for num in row:
                if num > threshold:
                    count += 1
        return count

    def replace_similar(original,restored,thresh):
        val = np.clip((original-restored),0,255)
        for i in range(np.shape(val)[0]):
            for j in range(np.shape(val)[1]):
                if val[i,j]<thresh:
                    restored[i,j] = original[i,j]
        return restored

    max_dim = np.shape(img)
    img_copy = np.zeros(max_dim)
    for i in range(0,int(max_dim[0]/window_dim[0])+1):
        for j in range(0,int(max_dim[1]/window_dim[1])+1):
            window = img[i*window_dim[0]:(i+1)*window_dim[0], j*window_dim[0]:(j+1)*window_dim[0]]
            window = erode_dilate_img(window,kernel,val_thresh) # Remove the text
            img_copy[i*window_dim[0]:(i+1)*window_dim[0], j*window_dim[0]:(j+1)*window_dim[0]] += window
    
    img_copy = cv2.erode(img_copy*1.00, kernel, iterations=1)
    img_copy = replace_similar(img,img_copy,100)
    img_copy = cv2.erode(img_copy*1.00, kernel, iterations=1)
    img_copy = replace_similar(img,img_copy,30)
    img_copy = cv2.erode(img_copy*1.00, kernel, iterations=1)
    img_copy = cv2.dilate(img_copy*1.00, kernel, iterations=3)
    img_copy = replace_similar(img,img_copy,20)
    img_copy[0,0] = 255

    return img_copy