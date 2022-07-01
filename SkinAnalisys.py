import cv2
import numpy as np

## 이미지 크기 설정
height = 320 #높이
width = 320 # 너비


## 픽셀단위로 접근하는 기능
def pixelDetector(img):
    cnt = 0
    for i in range(len(img)):
        for j in range(len(img[0])):
            if img[i][j] == 255:
                cnt += 1
    return cnt

# ^avr 기준 ^alpha 기울기
def contrastControlByHistogram(Img, standard, alpha):
    func = (1+alpha) * Img - (alpha * standard) 
    dst = np.clip(func, 0, 255).astype(np.uint8)
    return dst


# Canny Edge
## cv2.Canny(image, threshold1, threshold2, edge = None, apertureSize = None, L2gradient = None)
# Gausian Blur
## cv2.GaussianBlur(image, ksize, sigmaX, dst=None, sigmaY=None, borderType=None)
### [sigmaX, sigmaY : x, y 편향] 
### [ksize : 가우시안 커널 크기, (0, 0)을 지정하면 sigma 값에 의해 자동 결정됨]
### [borderType : 가장자리 픽셀 확장 방식]

def canny(img):
    canny = cv2.Canny(img, 150, 450) ## 2:1 혹은 3:1 의비율을 권함
    return canny

def deleteSkinTexture(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.inRange(img, 110, 255)
    return img



################# workspace #################

# Histogram
# cv2.calcHist(src, channels, mask, histSize, ranges, hist=None, accmulate=None)
# src: 입력 이미지의 리스트. 리스트 형태로 받기 때문에, 여러장의 이미지에 대해서도 히스토그램을 구할 수 있다.
# channels: 히스토그램을 구할 채널을 나타내는 리스트. GrayScale이라면 [0]을, BGR이라면 [0, 1, 2]가 된다.
# mask: 마스크 이미지. 마스크의 ROI만 히스토그램을 구할때 사용.
# histSize: 히스토그램 각 차원의 크기(bin)를 나타내는 리스트. 예를들어 [256]이라면 모든 GrayScale 픽셀값이 나타나고, [128]이라면 2개의 픽셀의 빈도가 더해진 형태가 된다.
# range: 히스토그램의 최소값, 최대값의 범위. GrayScale이라면 [0, 256]이다. 맨 마지막 256값은 포함이 안되며 255까지 표현된다.

## 명암 대비 조절 코드
# src = cv2.imread('lenna.bmp', cv2.IMREAD_GRAYSACLE)
# alpha = 1.0
# func = (1+alpha) * src - (alpha * 128)
# dst = np.clip(func, 0, 255).astype(np.uint8)

oilly_img = cv2.imread(f"oilly_model/oilly1.jpg" , cv2.IMREAD_COLOR)
# oilly_img = cv2.resize(oilly_img, dsize = (width, height))    
oilly_img = contrastControlByHistogram(oilly_img, 128, -0.1)
cv2.imshow("Result", oilly_img)
cv2.waitKey(0)

'''

'''

'''

testImg = cv2.resize(cv2.imread("11.jpg" , cv2.IMREAD_COLOR), dsize = (width, height))
canny1 = cv2.GaussianBlur(testImg, (0, 0), sigmaX=2, sigmaY=2)
canny2 = cv2.Canny(canny1, 10, 100)
canny3 = cv2.Canny(testImg, 10, 100)

testImg2 = cv2.resize(cv2.imread("12.jpg" , cv2.IMREAD_COLOR), dsize = (width, height))
canny21 = cv2.GaussianBlur(testImg2, (0, 0), sigmaX=2, sigmaY=2)
canny22 = cv2.Canny(canny21, 10, 100)
canny23 = cv2.Canny(testImg2, 10, 100)

cv2.imshow("orgin", testImg)
cv2.imshow("canny1", canny1)
cv2.imshow("canny2", canny2)
cv2.imshow("canny3", canny3)
print(f"canny3 detected totoal {pixelDetector(canny3)/102400.}")


cv2.imshow("orgin2", testImg2)
cv2.imshow("canny21", canny21)
cv2.imshow("canny22", canny22)
cv2.imshow("canny23", canny23)
print(f"canny23 detected totoal {pixelDetector(canny23)/102400.}")
cv2.waitKey(0)

'''