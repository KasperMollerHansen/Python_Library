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

def background_equalizer(img, x_thresh=0.4, y_thresh=0.4, scale = 3, thresh=190): # Function for equalizing the background of an image
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
    
    def check_neighbours(img, list_ij): # function for listing neigherbour pixels that is "1" in a list.
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

def apply_colors(img): # apply different colors to different interger values. Set 0 to black as a background
    colors = ['black', 'red', 'blue', 'purple','darkgreen', 'yellow', 'orange','cyan','magenta','lime']
    num_colors = len(colors)
    cmap = ListedColormap(colors)
    # Plot the image
    plt.imshow(np.where(img== 0, 0, img % (num_colors - 1) + 1), cmap=cmap, interpolation='nearest')
    plt.show()

def morphFilter(img, selem, function): # Morph filter that can be used to dilation, erosion and median filter
    # Add padding to the image
    s = selem.shape[1]//2 #pad width
    t = selem.shape[0]//2 #pad height
    img = np.pad(img, pad_width=((t,t),(s,s)))
    
    #Create empty output image
    imout = np.zeros(img.shape, dtype=np.uint8)
    
    #loop through all pixels except for padding
    for row in range(t,img.shape[0]-t): 
        for col in range(s,img.shape[1]-s):
            # extract pixels within structuring element
            se_tmp = img[row-t:row+t+1, col-s:col+s+1]
            #Select pixels in structure element and apply function (min, max, median)
            imout[row,col] = function(se_tmp[selem>0])
    #remove padding
    return imout[s:-s,t:-t]

def replace_similar(original,processed,thresh): # After a opening (erosion then dilation) we can replace similar pixels, with the orginal pixels
    restored = processed.copy()
    val = np.abs(original-restored)
    for i in range(np.shape(val)[0]):
        for j in range(np.shape(val)[1]):
            if val[i,j]<thresh:
                restored[i,j] = original[i,j]
    return restored

def create_disk_filter(im,radius): # Create a disk for filter usage
    n,m = np.shape(im)
    # Create an n x m matrix filled with zeros
    matrix = np.zeros((n, m))
    # Calculate center coordinates
    center_x = (n-1) / 2
    center_y = (m-1) / 2
    # Iterate through each cell of the matrix
    for i in range(n):
        for j in range(m):
            # Calculate distance from center
            distance = np.sqrt((i - center_x)**2 + (j - center_y)**2)
            # If distance is less than or equal to x, set value to 1
            if distance <= radius:
                matrix[i][j] = 1
    return matrix

def fft(im): #Fourier transform with frequency shift
    fourier = np.fft.fftshift(np.fft.fft2(im))
    fourier_dB = np.log10(abs(fourier)+np.finfo(float).eps)
    return fourier,fourier_dB
def ifft(im): #inverseFourier transform with frequency shift
    return abs(np.fft.ifft2(np.fft.ifftshift(im)))