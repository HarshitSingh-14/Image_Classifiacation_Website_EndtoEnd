import numpy as np
import pywt
import cv2



# Same form ipython 


def w2d(img , mode ='haar', level = 1):
    imArray = img
    # DataType Conversion 
    # Convert to gray-scale 
    imArray = cv2.cvtColor(imArray,cv2.COLOR_RGB2GRAY)
    # TO float -> more depth image
    imArray = np.float32(imArray)
    imArray  = imArray/255;
    
    #Compute coeff -> 
    coeff = pywt.wavedec2(imArray, mode , level=level)
    
    #Processing coefficient 
    coeffs_H=list(coeff)
    coeffs_H[0] *= 0;
    
    # reconstruction
    imArray_H=pywt.waverec2(coeffs_H, mode);
    imArray_H *= 255;
    imArray_H = np.uint8(imArray_H)

    
    return imArray_H