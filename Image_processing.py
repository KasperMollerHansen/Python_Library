# Import all necessary packages
import warnings


################################
def rgb2gray(rgb, coefficient = [0.2989, 0.5870 , 0.1141]):
    if sum(coefficient).round(4) != 1:
        warnings.warn("Coefficients doesn't add up to 1. Results may not be accurate.", UserWarning)
    
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = coefficient[0] * r + coefficient[1] * g + coefficient[2] * b

    return gray