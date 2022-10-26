import numpy as np
import cv2
#fit module
def getModel():
    npaClassifications = np.loadtxt("./data/classifications.txt", np.float32)
    npaFlattenedImages = np.loadtxt("./data/flattened_images.txt", np.float32) 
    npaClassifications = npaClassifications.reshape((npaClassifications.size, 1))# reshape numpy array to 1d, necessary to pass to call to train
    kNearest = cv2.ml.KNearest_create()                   # instantiate KNN object
    kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)
    return kNearest