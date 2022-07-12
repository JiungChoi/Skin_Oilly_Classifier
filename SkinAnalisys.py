import cv2
import numpy as np
from matplotlib import pyplot as plt


## 이미지 크기 설정
height = 320 #높이
width = 320 # 너비

## 유분측정 기준값 (광량)
LUX_INDEX = 140


## 픽셀단위로 접근
def pixelDetector(img):
    cnt = 0
    for i in range(len(img)):
        for j in range(len(img[0])):
            if img[i][j] == 255:
                cnt += 1
    return cnt

# 명도평준화(단일): ^avr 기준 ^alpha 기울기
def saturate_contrastA(Img, standard, alpha):
    func = (1+alpha) * Img - (alpha * standard) 
    dst = np.clip(func, 0, 255).astype(np.uint8)
    return dst

# 먕도 평준화(제어)
def saturate_contrastB(Img, boundary):
    I = printContrast(Img)
    
    Img = Img+(boundary - I)
    Img = np.clip(Img, 0, 255).astype(np.uint8)
    # print(f"{I} -> {printContrast(Img)}")
    return Img

# 이미지 명도 출력
def printContrast(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    sort_v = np.sort(v.flatten())
    mask = np.where(sort_v!=0)

    mask_sort_v = sort_v[mask]

    cut_v = mask_sort_v[int(width*height*0.05):int(width*height*0.95)]
    return np.median(cut_v)

def adjust_deadline_value(img, st, end, alpha=0.3):
    img_b, img_g, img_r = cv2.split(img)
    avrColor = (img_b.sum()/(width*height), img_g.sum()/(width*height), img_r.sum()/(width*height))

    img_b = np.clip(img_b, st, end).astype(np.uint8)
    img_g = np.clip(img_g, st, end).astype(np.uint8)
    img_r = np.clip(img_r, st, end).astype(np.uint8)

    img_b[np.where(img_b==st)] = 0
    img_g[np.where(img_g==st)] = 0
    img_r[np.where(img_r==st)] = 0
    
    img = cv2.merge((img_b, img_g, img_r))
    return img

def split_channer(img):
    img_b, img_g, img_r = cv2.split(img)
    img_empty = np.full_like(img_b, 0)

    mereged_img_b = cv2.merge((img_b, img_empty, img_empty))
    mereged_img_g = cv2.merge((img_empty, img_g, img_empty))
    mereged_img_r = cv2.merge((img_empty, img_empty, img_r))

    '''
    cv2.imshow("Chnnel_Blue", mereged_img_b)
    cv2.imshow("Chnnel_Green", mereged_img_g)
    cv2.imshow("Chnnel_Red", mereged_img_r)
    cv2.waitKey(0)
    
    cv2.destroyWindow("Chnnel_Blue")
    cv2.destroyWindow("Chnnel_Green")
    cv2.destroyWindow("Chnnel_Red")
    '''
    return mereged_img_b, mereged_img_g, mereged_img_r

    # cv2.destroyWindow(winname)

def output_img_merge_horizon(img1, img2, img3, img4):
    output_img = cv2.hconcat([cv2.hconcat([cv2.hconcat([img1, img2]), img3]) , img4])
    return output_img

def output_img_merge_vertical(img1, img2, img3, img4):
    output_img = cv2.vconcat([cv2.vconcat([cv2.vconcat([img1, img2]), img3]), img4])
    return output_img



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

def plot_histogram_bin(img):
    plt.hist(img.ravel(), 256, [0,256]); 
    plt.show()

def plot_histogram_rgb(img):
    print(img.shape)
    # 0 : Blue, 1: Green, 2: Red
    blue_hist = cv2.calcHist([img],[0],None,[256],[0,256])
    green_hist = cv2.calcHist([img],[1],None,[256],[0,256])
    red_hist = cv2.calcHist([img],[2],None,[256],[0,256])

    plt.subplot(111), 
    plt.plot(blue_hist , color='b'), 
    plt.plot(green_hist, color='g'), 
    plt.plot(red_hist, color='r'),
    plt.xlim([0,256])
    plt.show()

################# workspace #################

# Histogram
# cv2.calcHist(src, channels, mask, histSize, ranges, hist=None, accmulate=None)
# src: 입력 이미지의 리스트. 리스트 형태로 받기 때문에, 여러장의 이미지에 대해서도 히스토그램을 구할 수 있다.
# channels: 히스토그램을 구할 채널을 나타내는 리스트. GrayScale이라면 [0]을, BGR이라면 [0, 1, 2]가 된다.
# mask: 마스크 이미지. 마스크의 ROI만 히스토그램을 구할때 사용.
# histSize: 히스토그램 각 차원의 크기(bin)를 나타내는 리스트. 예를들어 [256]이라면 모든 GrayScale 픽셀값이 나타나고, [128]이라면 2개의 픽셀의 빈도가 더해진 형태가 된다.
# range: 히스토그램의 최소값, 최대값의 범위. GrayScale이라면 [0, 256]이다. 맨 마지막 256값은 포함이 안되며 255까지 표현된다.

# inRange
# cv2.inRange(src, lowerb, upperb, dst=None)
# src: 처리할 이미지
# lowerb : 하한값
# upperb : 상한값
# dst : ouput. 선언하지 않을 시 결과가 return 되므로 따로 변수를 정해주면 됨


## Cv2 창관리
# cv2.destroyWindow(winname) : winname에 해당하는 창을 닫음
# cv2.resize(img, dsize = (width, height)) : img의 사이즈 변경

## 이미지 선언부 : 유분기 4개 건조함 4개로 나눔 ### ### ### ### ### ### ### ### ###
# 원본이미지
oilly_imgs = [cv2.imread(f"oilly_model/oilly1.jpg" , cv2.IMREAD_COLOR),
              cv2.imread(f"oilly_model/oilly2.jpg" , cv2.IMREAD_COLOR),
              cv2.imread(f"oilly_model/oilly3.jpg" , cv2.IMREAD_COLOR),
              cv2.imread(f"oilly_model/oilly4.jpg" , cv2.IMREAD_COLOR)]

dry_imgs = [cv2.imread(f"oilly_model/dry1.jpg" , cv2.IMREAD_COLOR),
            cv2.imread(f"oilly_model/dry2.jpg" , cv2.IMREAD_COLOR),
            cv2.imread(f"oilly_model/dry3.jpg" , cv2.IMREAD_COLOR),
            cv2.imread(f"oilly_model/dry4.jpg" , cv2.IMREAD_COLOR)]

# 명도 평준화 이미지
oilly_imgs_saturate = []
dry_imgs_saturate = []

# 명도 양극화 이미지
oilly_imgs_polarization = []
dry_imgs_polarization = []

# 벡터이미지
oilly_imgs_analysis_vector = []
dry_imgs_analysis_vector = []

# 벡터이미지 GRAY
oilly_imgs_analysis_vector_BIN = []
dry_imgs_analysis_vector_BIN = []


## 모든 기준 이미지 평준화 : 기준값 140기준으로 평준화시킴 ### ### ### ### ### ### ### ### ###
for img in oilly_imgs:
    oilly_imgs_saturate.append(saturate_contrastB(img, 140))
    
for img in dry_imgs:
    dry_imgs_saturate.append(saturate_contrastB(img, 140))

## 한 이미지 내에서 명도 양극화 : 기준값 128 기준으로 양극화시킴 ### ### ### ### ### ### ### ### ###
for img in oilly_imgs_saturate:
    oilly_imgs_polarization.append(saturate_contrastA(img, 128, 0.5))
    
for img in dry_imgs_saturate:
    dry_imgs_polarization.append(saturate_contrastA(img, 128, 0.5))

## 양극화 시킨 이미지 상에서 평균값을 뽑아내어 평균값과 양극화시킨 이미지의 벡터를 구한다 ### ### ### ### ### ### ### ### ###
for img in oilly_imgs_polarization:
    output = img
    for _ in range(5): 
        output = saturate_contrastB(output, 0)
    for _ in range(2):
        output = saturate_contrastA(output, 50, 0.5)
    oilly_imgs_analysis_vector.append(output)
    
for img in dry_imgs_polarization:
    output = img
    for _ in range(5): 
        output = saturate_contrastB(output, 0)
    for _ in range(2):
        output = saturate_contrastA(output, 50, 0.5)
    for _ in range(3): 
        output = saturate_contrastB(output, 0)
    for _ in range(2):
        output = saturate_contrastA(output, 50, 0.5)
    for _ in range(1): 
        output = saturate_contrastB(output, 0)
    for _ in range(2):
        output = saturate_contrastA(output, 50, 0.5)
    dry_imgs_analysis_vector.append(output)

## 양극화시킨 벡터 이미지를 이진 이미지로 변환
for i, img in enumerate(oilly_imgs_analysis_vector):
    oilly_imgs_analysis_vector_BIN.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    
for i, img in enumerate(dry_imgs_analysis_vector):
    dry_imgs_analysis_vector_BIN.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

## 이미지 출력 :  ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
for i, img in enumerate(oilly_imgs):
    cv2.imshow(f"Oilly{i+1}", img)
    
for i, img in enumerate(dry_imgs):
    cv2.imshow(f"Dry{i+1}", img)

## 이미지 저장:  ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
for i, img in enumerate(oilly_imgs):
    cv2.imwrite(f"output_imgs/Oilly{i+1}.jpg", img)

for i, img in enumerate(dry_imgs):
    cv2.imwrite(f"output_imgs/Dry{i+1}.jpg", img)

## 이미지들 가로로 합침 ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
# 결과 : 원본이미지
output_img_horizon_oilly = output_img_merge_horizon(cv2.resize(oilly_imgs[0], dsize = (width, height)),
                cv2.resize(oilly_imgs[1], dsize = (width, height)),
                cv2.resize(oilly_imgs[2], dsize = (width, height)),
                cv2.resize(oilly_imgs[3], dsize = (width, height)))

output_img_horizon_dry = output_img_merge_horizon(cv2.resize(dry_imgs[0], dsize = (width, height)),
                cv2.resize(dry_imgs[1], dsize = (width, height)),
                cv2.resize(dry_imgs[2], dsize = (width, height)),
                cv2.resize(dry_imgs[3], dsize = (width, height)))

# # 결과 : 명도 평준화
# output_img_horizon_oilly_result_1 = output_img_merge_horizon(cv2.resize(oilly_imgs_saturate[0], dsize = (width, height)),
#                 cv2.resize(oilly_imgs_saturate[1], dsize = (width, height)),
#                 cv2.resize(oilly_imgs_saturate[2], dsize = (width, height)),
#                 cv2.resize(oilly_imgs_saturate[3], dsize = (width, height)))

# output_img_horizon_dry_result_1 = output_img_merge_horizon(cv2.resize(dry_imgs_saturate[0], dsize = (width, height)),
#                 cv2.resize(dry_imgs_saturate[1], dsize = (width, height)),
#                 cv2.resize(dry_imgs_saturate[2], dsize = (width, height)),
#                 cv2.resize(dry_imgs_saturate[3], dsize = (width, height)))

# # 결과 : 명도 양극화
# output_img_horizon_oilly_result_2 = output_img_merge_horizon(cv2.resize(oilly_imgs_polarization[0], dsize = (width, height)),
#                 cv2.resize(oilly_imgs_polarization[1], dsize = (width, height)),
#                 cv2.resize(oilly_imgs_polarization[2], dsize = (width, height)),
#                 cv2.resize(oilly_imgs_polarization[3], dsize = (width, height)))

# output_img_horizon_dry_result_2 = output_img_merge_horizon(cv2.resize(dry_imgs_polarization[0], dsize = (width, height)),
#                 cv2.resize(dry_imgs_polarization[1], dsize = (width, height)),
#                 cv2.resize(dry_imgs_polarization[2], dsize = (width, height)),
#                 cv2.resize(dry_imgs_polarization[3], dsize = (width, height)))


# 결과 : 이미지 벡터 분석 
output_img_horizon_oilly_result_3 = output_img_merge_horizon(cv2.resize(oilly_imgs_analysis_vector[0], dsize = (width, height)),
                cv2.resize(oilly_imgs_analysis_vector[1], dsize = (width, height)),
                cv2.resize(oilly_imgs_analysis_vector[2], dsize = (width, height)),
                cv2.resize(oilly_imgs_analysis_vector[3], dsize = (width, height)))

output_img_horizon_dry_result_3 = output_img_merge_horizon(cv2.resize(dry_imgs_analysis_vector[0], dsize = (width, height)),
                cv2.resize(dry_imgs_analysis_vector[1], dsize = (width, height)),
                cv2.resize(dry_imgs_analysis_vector[2], dsize = (width, height)),
                cv2.resize(dry_imgs_analysis_vector[3], dsize = (width, height)))

# 결과 : 이미지 벡터 분석(이진)
output_img_horizon_oilly_bin = output_img_merge_horizon(cv2.resize(oilly_imgs_analysis_vector_BIN[0], dsize = (width, height)),
                cv2.resize(oilly_imgs_analysis_vector_BIN[1], dsize = (width, height)),
                cv2.resize(oilly_imgs_analysis_vector_BIN[2], dsize = (width, height)),
                cv2.resize(oilly_imgs_analysis_vector_BIN[3], dsize = (width, height)))

output_img_horizon_dry_bin = output_img_merge_horizon(cv2.resize(dry_imgs_analysis_vector_BIN[0], dsize = (width, height)),
                cv2.resize(dry_imgs_analysis_vector_BIN[1], dsize = (width, height)),
                cv2.resize(dry_imgs_analysis_vector_BIN[2], dsize = (width, height)),
                cv2.resize(dry_imgs_analysis_vector_BIN[3], dsize = (width, height)))


# 커팅한 이미지들을 합침 : 수직 
output_img = output_img_merge_vertical(output_img_horizon_oilly, 
                                        output_img_horizon_oilly_result_3,
                                        output_img_horizon_dry,
                                        output_img_horizon_dry_result_3)

output_img_bin = output_img_merge_vertical(output_img_horizon_oilly, 
                                        output_img_horizon_oilly_bin,
                                        output_img_horizon_dry,
                                        output_img_horizon_dry_bin)






# 결과 이미지 출력  ### ### ### ### ### ### ### ### ###
# 
cv2.imshow("result", output_img)
cv2.imshow("result2", output_img_bin)


'''
oilly_img2 = contrastControlByHistogram(oilly_img1, LUX_INDEX, 0.3)
cv2.imshow("Result2", oilly_img2)
plot_histogram_bin(oilly_img2)


oilly_img3 = contrastControlByHistogram(oilly_img1, LUX_INDEX, 0.7)
cv2.imshow("Result3", oilly_img3)
plot_histogram_bin(oilly_img3)


oilly_img4 = contrastControlByHistogram(oilly_img1, LUX_INDEX, -0.3)
cv2.imshow("Result4", oilly_img4)
plot_histogram_bin(oilly_img4)

oilly_img5 = contrastControlByHistogram(oilly_img1, LUX_INDEX, -0.7)
cv2.imshow("Result5", oilly_img5)
plot_histogram_bin(oilly_img5)

cv2.imwrite("output.jpg", oilly_img1)
'''



'''
# 1차적으로 명도 차이를 줌 
contrastControled_img = contrastControlByHistogram(oilly_img1, 128, 0.3)
cv2.imshow("Result2", contrastControled_img)

# clip 함수를 이용해서 기준값으로 커팅함 (100, 170, 200)
output_img1 = adjust_deadline_value(contrastControled_img, 100, 255, alpha=0.1)
output_img2 = adjust_deadline_value(contrastControled_img, 170, 255, alpha=0.1)
output_img3 = adjust_deadline_value(contrastControled_img, 200, 255, alpha=0.1)

# 커팅한 이미지들을 합침 : 수평
output_img1_b, output_img1_g, output_img1_r =split_channer(output_img1)

output_img_horizon1 = output_img_merge_horizon(cv2.resize(output_img1_b, dsize = (width, height)),
                cv2.resize(output_img1_g, dsize = (width, height)),
                cv2.resize(output_img1_r, dsize = (width, height)),
                cv2.resize(output_img1, dsize = (width, height)))

output_img2_b, output_img2_g, output_img2_r =split_channer(output_img2)

output_img_horizon2 = output_img_merge_horizon(cv2.resize(output_img2_b, dsize = (width, height)),
                cv2.resize(output_img2_g, dsize = (width, height)),
                cv2.resize(output_img2_r, dsize = (width, height)),
                cv2.resize(output_img2, dsize = (width, height)))

output_img3_b, output_img3_g, output_img3_r =split_channer(output_img3)

output_img_horizon3 = output_img_merge_horizon(cv2.resize(output_img3_b, dsize = (width, height)),
                cv2.resize(output_img3_g, dsize = (width, height)),
                cv2.resize(output_img3_r, dsize = (width, height)),
                cv2.resize(output_img3, dsize = (width, height)))

# output_img_1, 2, 3 간의 차이를 분석함으로써 유분기를 검출해냄
weight1, weight2 = 0.5, 0.5

output_img_Analysis_b = output_img1_b - (weight1*output_img2_b + weight2*output_img3_b)
output_img_Analysis_g = output_img1_g - (weight1*output_img2_g + weight2*output_img3_g)
output_img_Analysis_r = output_img1_r - (weight1*output_img2_r + weight2*output_img3_r)
output_img_Analysis = output_img1 - (weight1*output_img2 + weight2*output_img3)
output_img_Analysis_horizon = output_img_merge_horizon(cv2.resize(output_img_Analysis_b, dsize = (width, height)),
                cv2.resize(output_img_Analysis_g, dsize = (width, height)),
                cv2.resize(output_img_Analysis_r, dsize = (width, height)),
                cv2.resize(output_img_Analysis, dsize = (width, height)))




cv2.imshow("Output", output_img)
cv2.imshow("Output2", output_img_Analysis)
plot_histogram_bin(output_img_Analysis)
'''
# plot_histogram_rgb(output_img_Analysis)
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