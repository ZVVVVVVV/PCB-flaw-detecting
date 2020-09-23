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
            returnVect[26 * i:26 * (i + 1)] = img[i]
    return returnVect


olist1 = []  # 短路
olist2 = []  # 线路凸起，缺失通孔
olist3 = []  # 多线，线路外斑点

ilist1 = []  # 断路
ilist2 = []  # 线路凹陷，线路端点焊盘缺失
ilist3 = []  # 少线，独立焊盘缺失
flawlist = []
f2 = open('svm.model', 'rb')
s2 = f2.read()
svm_model = pickle.loads(s2)

imgtest_in = cv.imread("08_test.jpg")
imgori_in = cv.imread("08.JPG")

# --------------------------偏移图像配准
img0 = imgtest_in
img = cv.cvtColor(img0, cv.COLOR_BGR2GRAY)
ret, img = cv.threshold(img, 120, 255, cv.THRESH_BINARY)
img = cv.GaussianBlur(img, (7, 7), 0)
edges = cv.Canny(img, 100, 200)

lines = cv.HoughLines(edges, 3, np.pi / 180, 500)

for rho, theta in lines[0]:
    print(theta / 0.01745)
    if 135 <= (theta / 0.01745) <= 180:
        angle = -(180 - theta / 0.01745)
    elif 0 <= (theta / 0.01745) <= 45:
        angle = theta / 0.01745
    elif 45 < (theta / 0.01745) < 90:
        angle = -(90 - (theta / 0.01745))
    elif 90 < (theta / 0.01745) < 135:
        angle = (theta / 0.01745) - 90

row, col = img.shape
img0 = cv.cvtColor(img0, cv.COLOR_BGR2RGB)
M = cv.getRotationMatrix2D((col / 2, row / 2), angle, 1)
img1 = cv.warpAffine(img0, M, (col, row))
img1 = cv.cvtColor(img1, cv.COLOR_BGR2RGB)
img_ori = cv.cvtColor(imgori_in, cv.COLOR_BGR2GRAY)
img_test = cv.cvtColor(img1, cv.COLOR_RGB2GRAY)
orb = cv.ORB_create()

kp1, des1 = orb.detectAndCompute(img_ori, None)
kp2, des2 = orb.detectAndCompute(img_test, None)

bf = cv.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)
print(len(matches))
matchesMask = [[0, 0] for i in range(len(matches))]

x_acc = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
y_acc = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
num_acc = 0

for i, (m, n) in enumerate(matches):
    if m.distance < 0.5 * n.distance:  # 两个特征向量之间的欧氏距离，越小表明匹配度越高。
        matchesMask[i] = [1, 0]
        pt1 = kp1[m.queryIdx].pt  # trainIdx    是匹配之后所对应关键点的序号，第一个载入图片的匹配关键点序号
        pt2 = kp2[m.trainIdx].pt  # queryIdx  是匹配之后所对应关键点的序号，第二个载入图片的匹配关键点序号
        y_acc[num_acc] = pt2[0] - pt1[0]
        x_acc[num_acc] = pt2[1] - pt1[1]

        num_acc = num_acc + 1
        print(i, pt1, pt2)
        if num_acc == 10:
            break

y_total = sum(y_acc) - min(y_acc) - max(y_acc)
x_total = sum(x_acc) - min(x_acc) - max(x_acc)

x_move = int(x_total / (num_acc - 2))
y_move = int(y_total / (num_acc - 2))
print(x_move)
print(y_move)
xy = img_ori.shape
print(xy)
get_ROI = img_test[x_move:(x_move + xy[0]), y_move:(y_move + xy[1])]
img = get_ROI  # 输出修正后图像的灰度图

# ---------------识别部分
img_ori_gray = cv.cvtColor(imgori_in, cv.COLOR_BGR2GRAY)
img_gray = get_ROI

# 二值化
ret2, th2 = cv.threshold(img_ori_gray, 45, 255, cv.THRESH_BINARY)
ret3, th3 = cv.threshold(img_gray, 45, 255, cv.THRESH_BINARY)

img_result1 = cv.bitwise_xor(th2, th3)
kernel_dilate = np.ones((3, 3), np.uint8)
kernel_erode = np.ones((3, 3), np.uint8)
img_result1 = cv.medianBlur(img_result1, 3)

# 开运算：对识别到的缺陷点边缘补偿，去噪
img_result1 = cv.bitwise_not(img_result1)
img_result1 = cv.erode(img_result1, kernel_erode, iterations=1)
img_result1 = cv.dilate(img_result1, kernel_dilate, iterations=1)

contours, para = cv.findContours(img_result1, cv.RETR_TREE, cv.CHAIN_APPROX_TC89_KCOS)

ret0, th0 = cv.threshold(img_ori_gray, 120, 255, cv.THRESH_BINARY)
ret1, th1 = cv.threshold(img_ori_gray, 160, 255, cv.THRESH_BINARY)
mask_out = cv.bitwise_xor(th0, th1)
mask_out[::] = 255
mask_out[110:(mask_out.shape[0] - 110), 110:(mask_out.shape[1] - 110)] = 0

mask_out = cv.bitwise_not(mask_out)
img_result1 = cv.bitwise_not(img_result1)
img_result3 = cv.bitwise_and(img_result1, mask_out)

kernel_erode = np.ones((3, 3), np.uint8)
img_result3 = cv.erode(img_result3, kernel_erode, iterations=2)
img_result3 = cv.dilate(img_result3, kernel_dilate, iterations=2)

oflaw = cv.bitwise_and(img_result3, th3, mask=img_result3)
iflaw = cv.bitwise_xor(img_result3, th3, mask=img_result3)

ocontours, opara = cv.findContours(oflaw, cv.RETR_TREE, cv.CHAIN_APPROX_TC89_KCOS)
icontours, ipara = cv.findContours(iflaw, cv.RETR_TREE, cv.CHAIN_APPROX_TC89_KCOS)

for i in range(len(ocontours)):
    th3_ori = copy.copy(th3)
    x, y, w, h = cv.boundingRect(ocontours[i])
    loc = [x, y, w, h]
    cv.drawContours(th3_ori, ocontours, i, (0, 0, 0), -1)  # 去除缺陷，将缺陷涂黑，检查连通域
    t_minus = th3_ori[y - 20:y + h + 20, x - 20:x + w + 20]
    kernel_erode = np.ones((3, 3), np.uint8)
    t_minus = cv.erode(t_minus, kernel_erode, iterations=1)
    t_o = th3[y - 20:y + h + 20, x - 20:x + w + 20]
    timg = oflaw[y - int(h / 3):y + h + int(h / 3), x - int(w / 3):x + w + int(w / 3)]
    oimg = cv.resize(timg, (32, 32))
    n_minus = cv.connectedComponents(t_minus)
    n_o = cv.connectedComponents(t_o)
    res = n_minus[0] - n_o[0]
    if res > 0:  # 短路
        flawlist.append([loc, "short"])
    elif res == 0:  # 线路凸起
        flawlist.append([loc, "spur"])
    else:  # 多线，线路外斑点
        flawlist.append([loc, "spurious copper"])

for i in range(len(icontours)):
    th3_ori = copy.copy(th3)
    x, y, w, h = cv.boundingRect(icontours[i])
    loc = [x, y, w, h]
    cv.drawContours(th3_ori, icontours, i, (255, 255, 255), -1)  # 去除缺陷，将缺陷涂白，检查连通域
    t_minus = th3_ori[y - 20:y + h + 20, x - 20:x + w + 20]
    t_o = th3[y - 20:y + h + 20, x - 20:x + w + 20]
    n_minus = cv.connectedComponents(t_minus)
    n_o = cv.connectedComponents(t_o)
    res = n_minus[0] - n_o[0]
    if res < 0:  # 断路
        flawlist.append([loc, "open circuit"])
    elif res == 0:  # 线路凹陷，线路端点焊盘缺失
        timg = img_gray[y - int(h / 3):y + h + int(h / 3), x - int(w / 3):x + w + int(w / 3)]
        oimg = cv.resize(timg, (26, 26))
        vec = []
        vec1 = img2vector(oimg)
        vec2 = img2vector(oimg)
        vec.append(vec1)
        vec.append(vec2)
        avec = np.array(vec)
        print(np.size(avec, 0), np.size(avec, 1))
        res = svm_model.predict(avec)
        if (res[0] == 0):  # 缺失通孔
            flawlist.append([loc, "missing hole"])
        elif (res[0] == 1):  # 线路凹陷
            flawlist.append([loc, "mouse bite"])
    else:  # 少线，独立焊盘缺失
        flawlist.append([loc, "loss line"])

lenth = len(flawlist)
for i in range(lenth):
    l = flawlist[i]
    [x, y, w, h] = l[0]
    name = l[1]
    cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3)
    cv.putText(img, name, (x - 6, y - 8), cv.FONT_HERSHEY_COMPLEX, 3, (255, 255, 255))

print(name)
plt.imshow(img, 'gray')
plt.show()
