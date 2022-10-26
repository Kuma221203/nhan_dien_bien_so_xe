import cv2
 
def imgEx(imgOriginal):
	imgHSV = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2HSV)
	imgHue, imgSat, imgValue = cv2.split(imgHSV)
	return imgValue

def maximizeContrast(imgGrayscale):
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
	topHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_TOPHAT, kernel, iterations = 10)
	blackHat =cv2.morphologyEx(imgGrayscale, cv2.MORPH_BLACKHAT, kernel, iterations = 10)
	imgGrayscalePlusTopHat = cv2.add(imgGrayscale, topHat)
	imgGraysclaePusTopHatMinusBlackHat = cv2.subtract(imgGrayscalePlusTopHat, blackHat)
	return imgGraysclaePusTopHatMinusBlackHat

def preprocess(imgOriginal):
	imgGrayscale = imgEx(imgOriginal)
	imgMaxContrast = maximizeContrast(imgGrayscale)
	imgBlur = cv2.GaussianBlur(imgMaxContrast, (5,5), 0)
	imgThresh = cv2.adaptiveThreshold(imgBlur, 255.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 9)
	imgCanny = cv2.Canny(imgThresh, 250, 255)
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
	imgDilate = cv2.morphologyEx(imgCanny, cv2.MORPH_DILATE, kernel, iterations = 1)
	contours, hierarchy = cv2.findContours(imgDilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]
	return imgGrayscale, imgThresh, contours, hierarchy
