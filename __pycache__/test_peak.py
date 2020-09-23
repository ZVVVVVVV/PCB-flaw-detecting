import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
def img2vector(img):
    returnVect = []
    for i in range(26):
        for j in range(26):
            returnVect[26*i:26*(i+1)] = img[i]
    return returnVect
ind_hole = [1, 4, 5, 6, 7, 8, 9]
data0 = []  # 缺失通孔
l0 = []  # label=0
for i in range(5):  # 得到缺失通孔的数据集
    img_ori = cv.imread("D:\PCB_DATASET\PCB_USED\\0%d.JPG" % (ind_hole[i]))
    img_ori_gray = cv.cvtColor(img_ori, cv.COLOR_BGR2GRAY)
    ret0, th2 = cv.threshold(img_ori_gray, 45, 255, cv.THRESH_BINARY_INV)
    for j in range(1, 10):
        img = cv.imread("D:\PCB_DATASET\images\Missing_hole\\0%d_missing_hole_0%d.jpg" % (ind_hole[i], j))
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret1, th3 = cv.threshold(img_gray, 45, 255, cv.THRESH_BINARY)

        img_result1 = cv.bitwise_xor(th2, th3)
        img_result2 = cv.bitwise_not(img_result1)
        kernel_dilate = np.ones((5, 5), np.uint8)
        kernel_erode = np.ones((5, 5), np.uint8)

       # 开运算：对识别到的缺陷点边缘补偿，顺带去个噪
        img_result1 = cv.bitwise_not(img_result1)
        img_result1 = cv.erode(img_result1, kernel_erode, iterations=1)
        img_result1 = cv.dilate(img_result1, kernel_dilate, iterations=1)
        # oflaw = cv.bitwise_and(img_result1, th3, mask=img_result1)
        iflaw = cv.bitwise_xor(img_result1, th3, mask=img_result1)
            # ocontours, opara = cv.findContours(oflaw, cv.RETR_TREE, cv.CHAIN_APPROX_TC89_KCOS)
        icontours, ipara = cv.findContours(iflaw, cv.RETR_TREE, cv.CHAIN_APPROX_TC89_KCOS)

        for k in range(len(icontours)):
            x, y, w, h = cv.boundingRect(icontours[k])
            timg = img_gray[y - int(h / 3):y + h + int(h / 3), x - int(w / 3):x + w + int(w / 3)]
            oimg = cv.resize(timg, (26, 26))
                # fimg= cv.equalizeHist(oimg)
                #plt.subplot(1, 2, 1), plt.imshow(timg, 'gray')
                #plt.subplot(1,2,2),plt.imshow(oimg, 'gray')
                #plt.show()
                # corners=cv.goodFeaturesToTrack(timg,  200,0.04,1)
            vec = img2vector(oimg)
            data0.append(vec)
            l0.append(0)

    # D:\PCB_DATASET\images\Mouse_bite
    # D:\PCB_DATASET\PCB_USED
ind_bite = [1, 4, 5, 6, 7, 8, 9]
data1 = []  # 线路缺陷
l1 = []  # label=1
for i in range(5):  # 得到线路缺陷的数据集
    img_ori = cv.imread("D:\PCB_DATASET\PCB_USED\\0%d.JPG" % (ind_bite[i]))
    img_ori_gray = cv.cvtColor(img_ori, cv.COLOR_BGR2GRAY)
    ret0, th2 = cv.threshold(img_ori_gray, 45, 255, cv.THRESH_BINARY_INV)
    for j in range(1, 10):
        img = cv.imread("D:\PCB_DATASET\images\Mouse_bite\\0%d_mouse_bite_0%d.jpg" % (ind_bite[i], j))
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret1, th3 = cv.threshold(img_gray, 45, 255, cv.THRESH_BINARY)

        img_result1 = cv.bitwise_xor(th2, th3)
        img_result2 = cv.bitwise_not(img_result1)
        kernel_dilate = np.ones((5, 5), np.uint8)
        kernel_erode = np.ones((5, 5), np.uint8)

            # 开运算：对识别到的缺陷点边缘补偿，顺带去个噪
        img_result1 = cv.bitwise_not(img_result1)
        img_result1 = cv.erode(img_result1, kernel_erode, iterations=1)
        img_result1 = cv.dilate(img_result1, kernel_dilate, iterations=1)
        # oflaw = cv.bitwise_and(img_result1, th3, mask=img_result1)
        iflaw = cv.bitwise_xor(img_result1, th3, mask=img_result1)

        icontours, ipara = cv.findContours(iflaw, cv.RETR_TREE, cv.CHAIN_APPROX_TC89_KCOS)

        for k in range(len(icontours)):
                # th3_ori = copy.copy(th3)
            x, y, w, h = cv.boundingRect(icontours[k])
            timg = img_gray[y - int(h / 3):y + h + int(h / 3), x - int(w / 3):x + w + int(w / 3)]
            oimg = cv.resize(timg, (26, 26))
                # plt.imshow(oimg,'gray')
                # plt.show()
                # corners=cv.goodFeaturesToTrack(timg,  200,0.04,1)
            vec = img2vector(oimg)
            data1.append(vec)
            l1.append(1)

data = data0
data.extend(data1)
adata = np.array(data)
l = l0
l.extend(l1)
al = np.array(l)
lenth = len(al)
labels = al.reshape(lenth, 1)
np.savetxt('data.txt', adata, fmt="%d") #保存为整数
np.savetxt('labels.txt', labels, fmt="%d") #保存为整数
tdata = np.loadtxt('data.txt')
print(np.size(tdata,0),np.size(tdata,1))