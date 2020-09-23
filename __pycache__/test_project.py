import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import copy

img_ori = cv.imread("04.JPG")
img = cv.imread("04_short_10t.jpg")
img_ori_gray = cv.cvtColor(img_ori,cv.COLOR_BGR2GRAY)
img_gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

# ---------------二值化试验
# -人工阈值√√ ----灰度图直方图，取第一个峰和第二个峰之间的谷值作阈值//PCB有三层亮度：焊盘、覆漆线路、覆漆底板
ret0,th2=cv.threshold(img_ori_gray,45,255,cv.THRESH_BINARY_INV)
ret1,th3=cv.threshold(img_gray,45,255,cv.THRESH_BINARY)
th3_ori = copy.copy(th3)
# ------------------------------------------------------------------------------------------------------
# ---大津法????
#ret2,th2 = cv.threshold(img_ori_gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
#ret3,th3 = cv.threshold(img_gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
# ------------------------------------------------------------------------------------------

img_result1 = cv.bitwise_xor(th2,th3)         #得到待测与标准的差异区域
#img_result2 = cv.bitwise_not(img_result1)
kernel_dilate = np.ones((5,5),np.uint8)
kernel_erode = np.ones((5,5),np.uint8)

#img_result1 = cv.medianBlur(img_result1,7)

#开运算：对识别到的缺陷点边缘补偿，顺带去个噪
img_result1 = cv.bitwise_not(img_result1)
img_result1 = cv.erode(img_result1,kernel_erode,iterations= 1)
img_result1 = cv.dilate(img_result1,kernel_dilate,iterations= 1)
t1 = cv.bitwise_and(img_result1,th3, mask=img_result1)
t2 = cv.bitwise_xor(img_result1,th3, mask=img_result1)
#在掩模下按位与，得到待测图像的多出而非缺陷区域。此结果在缺陷如短路位置边缘会出现细线，证明掩模过大，需进行调整

#img_gray = cv.cvtColor(img_gray,cv.COLOR_GRAY2BGR)
#img_gray0 = img_gray
img_ori_gray = cv.cvtColor(img_ori_gray,cv.COLOR_GRAY2BGR)
contours,para = cv.findContours(t2, cv.RETR_TREE, cv.CHAIN_APPROX_TC89_KCOS)
img_result3 =cv.drawContours(img_gray, contours, -1, (0,0,255),3)

'''
print(len(contours))
plt.subplot(2, 3, 1), plt.imshow(img_ori_gray, 'gray'), plt.title("img_ori_gray")
plt.subplot(2, 3, 2), plt.imshow(img_gray0, 'gray'), plt.title("img_gray")
plt.subplot(2, 3, 3), plt.imshow(img, 'gray'), plt.title("img_ori_gray")
plt.subplot(2, 3, 4), plt.imshow(img_result3, 'gray'), plt.title("img_gray")
plt.subplot(2, 3, 5),plt.imshow(t1,'gray'),plt.title("img_result")
plt.subplot(2, 3, 6),plt.imshow(th3,'gray'),plt.title("img_result")
plt.show()
'''
#cnt=contours[2]
x,y,w,h = cv.boundingRect(contours[2])

#temp=cv.drawContours(th3, contours, 2, (0,0,255),3)
im_temp=cv.drawContours(th3_ori, contours, 0 , (255,255,255),-1)
hole=t2[y - int(h/3):y + h + int(h/3), x - int(w/3):x + w + int(w/3)]
oimg = cv.resize(hole, (32, 32))
fast = cv.FastFeatureDetector_create()

# find and draw the keypoints
kp = fast.detect(hole,None)
img2=copy.copy(hole)
cv.drawKeypoints(hole, kp, img2 , color=(255,0,0))
c1,cpara=cv.findContours(hole, cv.RETR_TREE, cv.CHAIN_APPROX_TC89_KCOS)


maxcorners = 200

qualityLevel = 0.01 # 角点检测可接受的最小特征值

minDistance = 10 # 角点之间最小距离

blockSize = 3 #计算导数自相关矩阵时指定的领域范围

k = 0.04 # 权重系数

corners=cv.goodFeaturesToTrack(hole,  200,0.04,1)
print(np.size(corners))
#cv.drawKeypoints(hole, c1, img2 , color=(255,0,0))
plt.subplot(1, 3, 1), plt.imshow(th2, 'gray'), plt.title("temp")
plt.subplot(1, 3, 2), plt.imshow(th3, 'gray'), plt.title("im_temp")
plt.subplot(1, 3, 3), plt.imshow(img2, 'gray'), plt.title("th2")
plt.show()

'''
t_minus=im_temp[y-50:y+h+50,x-50:x+w+50]
t_o=th3[y-50:y+h+50,x-50:x+w+50]
n_minus=cv.connectedComponents(t_minus)
n_o=cv.connectedComponents(t_o)
res=n_minus[0]-n_o[0]
print(n_minus[0],n_o[0])
print(x,y,w,h)
img = cv.rectangle(th3_ori,(x-50,y-50),(x+w+50,y+h+50),(0,255,0),2)
cv.namedWindow('1', 0)
cv.imshow('1',img)
cv.waitKey(0)

'''






