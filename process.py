import cv2
import numpy as np
import Preprocess
import math
#quan la toi
# import char_license_plate_recognition

npaClassifications = np.loadtxt("classifications.txt", np.float32)
npaFlattenedImages = np.loadtxt("flattened_images.txt", np.float32) 
npaClassifications = npaClassifications.reshape((npaClassifications.size, 1))# reshape numpy array to 1d, necessary to pass to call to train
kNearest = cv2.ml.KNearest_create()                   # instantiate KNN object
kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)

Min_char = 0.01
Max_char = 0.09

Min_ratio_char = 0.25
Max_ratio_char = 0.7

RESIZE_IMG_WIDTH = 20
RESIZE_IMG_HEIGHT = 30

img = cv2.imread('img.jpg')
imgGrayscale, imgThresh, contours, hierarchy = Preprocess.preprocess(img)
screenCnt = []
for c in contours:
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.06*peri, True)
	[x, y, w, h] = cv2.boundingRect(approx.copy())
	if len(approx) == 4:
		screenCnt.append(approx)

if screenCnt is None:
	detected = 0
	print('None license plates')
else:
	for c in screenCnt:
		(x1, y1) = c[0, 0]
		(x2, y2) = c[1, 0]
		(x3, y3) = c[2, 0]
		(x4, y4) = c[3, 0]
		array = [[x1, y1],[x2, y2],[x3, y3],[x4, y4]]
		array.sort(reverse = True, key = lambda x:x[1])
		[x1, y1] = array[0]
		[x2, y2] = array[1]
		tan = abs(y1-y2)/abs(x1-x2)
		angle = math.atan(tan)*(180.0/math.pi)

		#cut picture
			#note: Contours save position, don't save matrix so we must convert between position to matrix ((x,y) -> (colomn, row))
		array = np.squeeze(c)
		(xtop, ytop) = (np.min(array[:,1]), np.min(array[:,0]))
		(xbottom, ybottom) = (np.max(array[:,1]), np.max(array[:,0]))
		imgThreshPlate = imgThresh[xtop:xbottom,ytop:ybottom]
		roi = img[xtop:xbottom,ytop:ybottom]
		plateCenter = (xbottom-xtop)/2 ,(ybottom-ytop)/2
		if x1<x2:
			rotaionMatrix = cv2.getRotationMatrix2D(plateCenter, -angle, 1) # it will be return 2x3 matrix
		else:
			rotaionMatrix = cv2.getRotationMatrix2D(plateCenter, angle, 1)
		imgPlate = cv2.warpAffine(imgThreshPlate, rotaionMatrix, (ybottom-ytop, xbottom-xtop))
		roi = cv2.warpAffine(roi, rotaionMatrix, (ybottom-ytop, xbottom-xtop))

		kernel_3 =  cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
		thres_dil = cv2.morphologyEx(imgPlate, cv2.MORPH_DILATE, kernel_3)
		cont, hie = cv2.findContours(thres_dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		# cont = sorted(cont,key = cv2.contourArea, reverse = True)[:10]
		char_x = []
		char_x_ind = {}
		height, width ,_ = roi.shape
		roiarea = height*width
		for ind,cnt in enumerate(cont):
			area = cv2.contourArea(cont[ind])
			(x,y,w,h) = cv2.boundingRect(cnt)
			ratiochar = w/h
			if(roiarea*Min_char < area < roiarea*Max_char) and (Min_ratio_char < ratiochar < Max_ratio_char):
				if x in char_x:
					x = x + 1
				char_x.append(x)
				char_x_ind[x] = ind
		if len(char_x) in range(7,10):
			char_x = sorted(char_x)
			stringFinal = ""
			firstStr = ""
			secondStr = ""

			for i in char_x:
				(x,y,w,h) = cv2.boundingRect(cont[char_x_ind[i]])
				cv2.rectangle(roi, (x,y), (x+w, y+h), (0, 255, 0), 2)
				imgRoi = imgPlate[y:y+h, x:x+w]
				imgRoi = cv2.resize(imgRoi, (RESIZE_IMG_WIDTH, RESIZE_IMG_HEIGHT))
				imgRoi = imgRoi.reshape((1, RESIZE_IMG_HEIGHT*RESIZE_IMG_WIDTH))
				imgRoi = np.float32(imgRoi)
				_, result, neigh_resp, dist = kNearest.findNearest(imgRoi, k = 3)
				strCurrentChar = str(chr(int(result[0][0])))
				if(y < height/3):
					firstStr += strCurrentChar
				else:
					secondStr += strCurrentChar
			stringFinal = firstStr + secondStr
			print(stringFinal)
		else:
			print('None license plate')
