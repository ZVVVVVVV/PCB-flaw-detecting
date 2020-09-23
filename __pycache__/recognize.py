import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import copy
import pickle
from sklearn import svm

def img2vector(img):
    returnVect = []
    for i in range(26):
        for j in range(26):
            returnVect[26*i:26*(i+1)] = img[i]
    return returnVect

olist1=[]    #短路
olist2=[]    #线路凸起，缺失通孔
olist3=[]    #多线，线路外斑点



ilist1=[]    #断路
ilist2=[]    #线路凹陷，线路端点焊盘缺失
ilist3=[]    #少线，独立焊盘缺失
flawlist=[]
f2=open('svm.model','rb')
s2=f2.read()
svm_model=pickle.loads(s2)

img_ori = cv.imread("04.JPG")
img = cv.imread("04_short_10t.jpg")
img_ori_gray = cv.cvtColor(img_ori, cv.COLOR_BGR2GRAY)
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# ---------------二值化试验
# -人工阈值√√ ----灰度图直方图，取第一个峰和第二个峰之间的谷值作阈值//PCB有三层亮度：焊盘、覆漆线路、覆漆底板
ret0, th2 = cv.threshold(img_ori_gray, 45, 255, cv.THRESH_BINARY)
ret1, th3 = cv.threshold(img_gray, 45, 255, cv.THRESH_BINARY)
#h3_ori = copy.copy(th3)
# ------------------------------------------------------------------------------------------------------
# ---大津法????
#ret2,th2 = cv.threshold(img_ori_gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
#ret3,th3 = cv.threshold(img_gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
# ------------------------------------------------------------------------------------------

img_result1 = cv.bitwise_xor(th2, th3)

img_result2 = cv.bitwise_not(img_result1)
kernel_dilate = np.ones((3, 3), np.uint8)
kernel_erode = np.ones((3, 3), np.uint8)


#开运算：对识别到的缺陷点边缘补偿，顺带去个噪

img_result1 = cv.erode(img_result1, kernel_erode, iterations=1)
img_result1 = cv.dilate(img_result1, kernel_dilate, iterations=1)


oflaw = cv.bitwise_and(img_result1, th3, mask=img_result1)
iflaw = cv.bitwise_xor(img_result1, th3, mask=img_result1)

plt.subplot(141),plt.imshow(iflaw)
plt.subplot(142),plt.imshow(oflaw)
plt.subplot(143),plt.imshow(img_result1)
plt.subplot(144),plt.imshow(th3),plt.show
#img_gray = cv.cvtColor(img_gray,cv.COLOR_GRAY2BGR)
#img_gray0 = img_gray
#img_ori_gray = cv.cvtColor(img_ori_gray,cv.COLOR_GRAY2BGR)
ocontours,opara = cv.findContours(oflaw, cv.RETR_TREE, cv.CHAIN_APPROX_TC89_KCOS)
icontours,ipara = cv.findContours(iflaw, cv.RETR_TREE, cv.CHAIN_APPROX_TC89_KCOS)
#img_result3 =cv.drawContours(img_gray, ocontours, -1, (0,0,255),3)

for i in range(len(ocontours)):
    th3_ori = copy.copy(th3)
    x, y, w, h = cv.boundingRect(ocontours[i])
    loc = [x, y, w, h]
    cv.drawContours(th3_ori, ocontours, i, (0, 0, 0), -1)   #去除缺陷，将缺陷涂黑，检查连通域
    t_minus = th3_ori[y - 20:y + h + 20, x - 20:x + w + 20]
    kernel_erode = np.ones((3,3), np.uint8)
    t_minus = cv.erode(t_minus, kernel_erode, iterations=1)
    t_o = th3[y - 20:y + h + 20, x - 20:x + w + 20]
    timg = oflaw[y - int(h/3):y + h + int(h/3), x - int(w/3):x + w + int(w/3)]
    oimg=cv.resize(timg, (32,32))
    n_minus = cv.connectedComponents(t_minus)
    n_o = cv.connectedComponents(t_o)
    res = n_minus[0] - n_o[0]
    if res>0:     #短路
        flawlist.append([loc,"short"])
    elif res == 0:     #线路凸起
        flawlist.append([loc,"spur"])
    else:        #多线，线路外斑点
        flawlist.append([loc,"spurious copper"])

for i in range(len(icontours)):
    th3_ori = copy.copy(th3)
    x,y,w,h = cv.boundingRect(icontours[i])
    loc=[x,y,w,h]
    cv.drawContours(th3_ori, icontours, i, (255, 255, 255), -1)     #去除缺陷，将缺陷涂白，检查连通域
    t_minus = th3_ori[y - 20:y + h + 20, x - 20:x + w + 20]
    t_o = th3[y - 20:y + h + 20, x - 20:x + w + 20]
    #timg = img_gray[y - int(h/3):y + h + int(h/3), x - int(w/3):x + w + int(w/3)]
    #oimg=cv.resize(timg, (32, 32))
    n_minus = cv.connectedComponents(t_minus)
    n_o = cv.connectedComponents(t_o)
    res = n_minus[0] - n_o[0]
    if res<0:     #断路
        flawlist.append([loc,"open circuit"])
    elif res == 0:     #线路凹陷，线路端点焊盘缺失
        timg = img_gray[y - int(h / 3):y + h + int(h / 3), x - int(w / 3):x + w + int(w / 3)]
        oimg = cv.resize(timg, (26, 26))
        vec=[]
        vec1=img2vector(oimg)
        vec2 = img2vector(oimg)
        vec.append(vec1)
        vec.append(vec2)
        #data=np.array(vec)
        #data.e
        avec=np.array(vec)
        print(np.size(avec,0),np.size(avec,1))
        res=svm_model.predict(avec)
        if(res[0]==0):   #缺失通孔
            flawlist.append([loc,"missing hole"])
        elif(res[0]==1):  #线路凹陷
            flawlist.append([loc, "mouse bite"])
    else:        #少线，独立焊盘缺失
        flawlist.append([loc,"少线"])

lenth=len(flawlist)
for i in range(lenth):
    l=flawlist[i]
    [x,y,w,h]=l[0]
    name=l[1]
    cv.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 3)
    cv.putText(img, name, (x-6, y-8), cv.FONT_HERSHEY_COMPLEX, 3, (255, 255, 255))

print(name)
#plt.imshow(img, 'gray')
plt.show()