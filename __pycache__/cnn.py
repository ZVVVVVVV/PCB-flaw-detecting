import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import copy

olist1=[]    #短路
olist2=[]    #线路凸起，缺失通孔
olist3=[]    #多线，线路外斑点



ilist1=[]    #断路
ilist2=[]    #线路凹陷，线路端点焊盘缺失
ilist3=[]    #少线，独立焊盘缺失


img_ori = cv.imread("04.JPG")
img = cv.imread("04_short_10t.jpg")
img_ori_gray = cv.cvtColor(img_ori, cv.COLOR_BGR2GRAY)
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# ---------------二值化试验
# -人工阈值√√ ----灰度图直方图，取第一个峰和第二个峰之间的谷值作阈值//PCB有三层亮度：焊盘、覆漆线路、覆漆底板
ret0, th2=cv.threshold(img_ori_gray, 45, 255, cv.THRESH_BINARY_INV)
ret1, th3=cv.threshold(img_gray, 45, 255, cv.THRESH_BINARY)
#h3_ori = copy.copy(th3)
# ------------------------------------------------------------------------------------------------------
# ---大津法????
#ret2,th2 = cv.threshold(img_ori_gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
#ret3,th3 = cv.threshold(img_gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
# ------------------------------------------------------------------------------------------

img_result1 = cv.bitwise_xor(th2,th3)

img_result2 = cv.bitwise_not(img_result1)
kernel_dilate = np.ones((3,3),np.uint8)
kernel_erode = np.ones((3,3),np.uint8)

#img_result1 = cv.medianBlur(img_result1,7)

#开运算：对识别到的缺陷点边缘补偿，顺带去个噪
img_result1 = cv.bitwise_not(img_result1)
img_result1 = cv.erode(img_result1,kernel_erode,iterations= 1)
img_result1 = cv.dilate(img_result1,kernel_dilate,iterations= 1)



oflaw = cv.bitwise_and(img_result1,th3, mask=img_result1)
iflaw = cv.bitwise_xor(img_result1,th3, mask=img_result1)

#img_gray = cv.cvtColor(img_gray,cv.COLOR_GRAY2BGR)
#img_gray0 = img_gray
#img_ori_gray = cv.cvtColor(img_ori_gray,cv.COLOR_GRAY2BGR)
ocontours,opara = cv.findContours(oflaw, cv.RETR_TREE, cv.CHAIN_APPROX_TC89_KCOS)
icontours,ipara = cv.findContours(iflaw, cv.RETR_TREE, cv.CHAIN_APPROX_TC89_KCOS)
#img_result3 =cv.drawContours(img_gray, ocontours, -1, (0,0,255),3)

for i in range(len(ocontours)):
    th3_ori = copy.copy(th3)
    x,y,w,h = cv.boundingRect(ocontours[i])
    loc=[x,y,w,h]
    cv.drawContours(th3_ori, ocontours, i, (0, 0, 0), -1)   #去除缺陷，将缺陷涂黑，检查连通域
    t_minus = th3_ori[y - 20:y + h + 20, x - 20:x + w + 20]
    kernel_erode = np.ones((3,3), np.uint8)
    t_minus = cv.erode(t_minus, kernel_erode, iterations=1)
    t_o = th3[y - 20:y + h + 20, x - 20:x + w + 20]
    timg = oflaw[y - int(h/3):y + h + int(h/3), x - int(w/3):x + w + int(w/3)]
    oimg=cv.resize(timg, (32, 32))
    n_minus = cv.connectedComponents(t_minus)
    n_o = cv.connectedComponents(t_o)
    res = n_minus[0] - n_o[0]
    if res > 0:     # 短路
        olist1.append(t_minus)
    elif res == 0:     # 线路凸起，缺失通孔
        olist2.append(t_minus)
    else:        # 多线
        olist3.append(oimg)

for i in range(len(icontours)):
    th3_ori = copy.copy(th3)
    x, y, w, h = cv.boundingRect(icontours[i])
    loc = [x, y, w, h]
    cv.drawContours(th3_ori, icontours, i, (255, 255, 255), -1)     #去除缺陷，将缺陷涂白，检查连通域
    t_minus = th3_ori[y - 20:y + h + 20, x - 20:x + w + 20]
    t_o = th3[y - 20:y + h + 20, x - 20:x + w + 20]
    timg = img_gray[y - int(h/3):y + h + int(h/3), x - int(w/3):x + w + int(w/3)]
    oimg = cv.resize(timg, (32, 32))
    n_minus = cv.connectedComponents(t_minus)
    n_o = cv.connectedComponents(t_o)
    res = n_minus[0] - n_o[0]
    if res<0:     #断路
        ilist1.append(oimg)
    elif res == 0:     #线路凹陷，线路端点焊盘缺失
        ilist2.append(oimg)
    else:        #少线，独立焊盘缺失
        ilist3.append(oimg)

plt.subplot(1, 3, 1), plt.imshow(ilist2[0], 'gray'), plt.title("i2")
plt.subplot(1, 3, 2), plt.imshow(ilist2[1], 'gray'), plt.title("i2")
plt.subplot(1, 3, 3), plt.imshow(ilist2[0], 'gray'), plt.title("i2")
plt.show()
#print(ilist3)
#test = olist1[0]

#print(test[20])

