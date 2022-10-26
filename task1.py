# step1: chuyển sang ảnh xám
# step2: xử lí hình thái học (phép nở, phép co, tophat, blackhat)
# step3: làm mờ
# step4: lọc ngưỡng
# step5: phát hiện cạnh sử dụng Canny
# step6: tìm viền, lấy n viền có areaCountour lớn nhất
# step7: xấp xỉ đa giác, chọn đa giác có 4 đỉnh
# step8: xoay ảnh
# step9: tìm vùng kí tự (chọn các vùng viền có tỉ lệ cao rộng và và tỉ lệ diện tích phù hợp), và nhận diện ki tự
import cv2
import numpy as np
import math

SIZE_KERNEL = (3,3)
SIZE_BLUR = (5,5)

Min_char = 0.01
Max_char = 0.09

Min_ratio_char = 0.25
Max_ratio_char = 0.7

RESIZE_IMG_WIDTH = 20
RESIZE_IMG_HEIGHT = 30

npaClassifications = np.loadtxt("classifications.txt", np.float32)
npaFlattenedImages = np.loadtxt("flattened_images.txt", np.float32) 
npaClassifications = npaClassifications.reshape((npaClassifications.size, 1))# reshape numpy array to 1d, necessary to pass to call to train
kNearest = cv2.ml.KNearest_create()                   # instantiate KNN object
kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)

#step1:

def imgEx(img):
	imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	imgHue, imgSat, imgGrayValue = cv2.split(imgHSV)
	return imgGrayValue

#step2
def maximaizeContrast(imgGrayValue):
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, SIZE_KERNEL)
	imgTopHat = cv2.morphologyEx(imgGrayValue, cv2.MORPH_TOPHAT, kernel,iterations = 10)
	imgBlackHat = cv2.morphologyEx(imgGrayValue, cv2.MORPH_BLACKHAT, kernel, iterations = 10)
	imgMorph = cv2.subtract(cv2.add(imgGrayValue, imgTopHat), imgBlackHat)
	return imgMorph

#step3
def imgBlur(img):
	imgBlur = cv2.GaussianBlur(img, SIZE_BLUR, 0)
	return imgBlur

#step4
def imgThresh(img):
	imgThresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 9)
	return imgThresh

#step5
def getEdge(img):
	imgCanny = cv2.Canny(img, 250, 255)
	return imgCanny 

#step6
def getContours(img):
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, SIZE_KERNEL)
	imgDilate = cv2.morphologyEx(img, cv2.MORPH_DILATE, kernel, iterations = 1)
	contours, hierarchy = cv2.findContours(imgDilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
	contours = sorted(contours, key = cv2.contourArea , reverse = True)[:10]
	return contours

#step7
def getRect(contours):
	result = [] 
	for cnt in contours:
		primeter = cv2.arcLength(cnt, True)
		approx = cv2.approxPolyDP(cnt, primeter*0.06, True)
		if(len(approx) == 4):
			result.append(approx)
	numScreen = len(result)
	return result

#step8
def rotationImg(screenCnts, imgThresh):
	result = []
	if(screenCnts is None):
		print('No license plate')
	else:
		for cnt in screenCnts:
			cnt = np.squeeze(cnt)
			ml = cnt.tolist()
			ml.sort(key = lambda x:x[1], reverse = True)
			(x1,y1) = (ml[0][0], ml[0][1]) #coordinate
			(x2,y2) = (ml[1][0], ml[1][1]) #coordinate
			angle = math.atan(abs(y1 - y2)/abs(x1 - x2))*(180.0/math.pi)
			topLeft = (np.min(cnt[:,0]), np.min(cnt[:,1])) #coordinate
			bottomRight = (np.max(cnt[:,0]), np.max(cnt[:,1])) #coordinate
			imgLiPl = imgThresh[topLeft[1]:bottomRight[1],topLeft[0]:bottomRight[0]]
			plateCenter = ((bottomRight[0]-topLeft[0])/2,(bottomRight[1]-topLeft[1])/2)
			if(x1 < x2):
				rotationMatrix = cv2.getRotationMatrix2D(plateCenter, -angle, 1)
			else:
				rotationMatrix = cv2.getRotationMatrix2D(plateCenter, angle, 1)
			imgRotationLiPl = cv2.warpAffine(imgLiPl, rotationMatrix, (bottomRight[0]-topLeft[0],bottomRight[1]-topLeft[1]))
			result.append(imgRotationLiPl)
	return result

#step9
def findCharacter(imgLiPls):
	detect = 0
	for imgLiPl in imgLiPls:
		contours, hierarchy = cv2.findContours(imgLiPl, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		height,width = imgLiPl.shape
		char_x = []
		char_ind_x = {}
		area = height*width
		for ind,cnt in enumerate(contours):
			char_area = cv2.contourArea(cnt)
			(x,y,w,h) = cv2.boundingRect(cnt)
			char_ratio = w/h
			if(Min_char*area < char_area < Max_char*area) and (Min_ratio_char < char_ratio < Max_ratio_char):
				if(x in char_x ):
					x = x + 1
				char_x.append(x)
				char_ind_x[x] = ind
		if(6 < len(char_x) < 11):
			char_x = sorted(char_x)
			first_string = ""
			second_string = ""
			final_string = ""
			for i in char_x:
				(x,y,w,h) = cv2.boundingRect(contours[char_ind_x[i]])
				imgChar = imgLiPl[y:(y+h), x:(x+w)]
				imgCharRota = cv2.resize(imgChar, (RESIZE_IMG_WIDTH, RESIZE_IMG_HEIGHT))
				dataChar = imgCharRota.reshape(1, RESIZE_IMG_WIDTH*RESIZE_IMG_HEIGHT)
				dataChar = np.float32(dataChar)
				_,result,_,_ = kNearest.findNearest(dataChar, k = 3)
				charCurrent = str(chr(int(result[0][0])))
				if(y < height/3):
					first_string += charCurrent
				else:
					second_string += charCurrent
			final_string = first_string + second_string
			print(final_string)
			detect = detect + 1
	if detect == 0:
		print('No find license plate')

img = cv2.imread('./img/img2.jpg')
imgGray = imgEx(img)
imgMorph = maximaizeContrast(imgGray)
imgBlur = imgBlur(imgMorph)
imgThresh = imgThresh(imgBlur)
imgEdge = getEdge(imgThresh)
contours = getContours(imgEdge)
rect = getRect(contours)
imgRotaLiPls = rotationImg(rect, imgThresh)
findCharacter(imgRotaLiPls)