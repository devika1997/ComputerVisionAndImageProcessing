# 1. Only add your code inside the function (including newly improted packages). 
#  You can design a new function and call the new function in the given functions. 
# 2. For bonus: Give your own picturs. If you have N pictures, name your pictures such as ["t3_1.png", "t3_2.png", ..., "t3_N.png"], and put them inside the folder "images".
# 3. Not following the project guidelines will result in a 10% reduction in grades

import cv2
import numpy as np
import matplotlib.pyplot as plt
import json


def Overlap(img1,img2):
    olap=0
    matches1=[]
    matches2=[]
    keypoints1, descriptors1 = cv2.SIFT_create(450).detectAndCompute(img1,None)
    keypoints2, descriptors2 = cv2.SIFT_create(450).detectAndCompute(img2,None)
    for i in range(len(descriptors1)):
        for j in range(len(descriptors2)):
           dist=np.linalg.norm(descriptors1[i]-descriptors2[j])/len(descriptors1)
           if(dist<0.3):
                matches1.append(keypoints1[i].pt)
                matches2.append(keypoints2[j].pt)
    if len(matches1)>40:
        m1=np.array(matches1).reshape(-1,1,2)
        m2=np.array(matches2).reshape(-1,1,2)
        retval,_ = cv2.findHomography(m1, m2, cv2.RANSAC, 5)
        
        #images width and height
        h1,w1,_=img1.shape
        h2,w2,_=img2.shape

        tmpwp1 = np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2)
        wp2 = np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)
        wp1= cv2.perspectiveTransform(tmpwp1, retval)
        w = np.concatenate((wp1,wp2), axis=0)

        
        (l, t),(r, b) = np.int32(w.min(axis=0).ravel()), np.int32(w.max(axis=0).ravel())
        translation = np.array([[1.0, 0.0, -l], [0.0, 1.0, -t], [0.0, 0.0, 1.0]])
        warp_img1 = cv2.warpPerspective(img1, translation.dot(retval), (r-l, b-t))
        warp_img2 = cv2.warpPerspective(img2, translation,(r-l, b-t))

        img1_gray = cv2.cvtColor(warp_img1, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(warp_img2, cv2.COLOR_BGR2GRAY)
        i1=cv2.threshold(img1_gray, 1,255, cv2.THRESH_BINARY)[1]
        i2=cv2.threshold(img2_gray, 1,255, cv2.THRESH_BINARY)[1]
        result=cv2.bitwise_and(i1,i2)
        r=np.where(result>0,1,0)
        olap=(np.sum(r)/(i1.shape[0]*i1.shape[1]))+2
    return olap*100


def stitch_background(img1, img2):
    "The output image should be saved in the savepath."
    "Do NOT modify the code provided."
    matches1=[]
    matches2=[]
    keypoints1, descriptors1 = cv2.SIFT_create(450).detectAndCompute(img1,None)
    keypoints2, descriptors2 = cv2.SIFT_create(450).detectAndCompute(img2,None)
    for i in range(len(descriptors1)):
        for j in range(len(descriptors2)):
           dist=np.linalg.norm(descriptors1[i]-descriptors2[j])/len(descriptors1)
           if(dist<0.5):
                matches1.append(keypoints1[i].pt)
                matches2.append(keypoints2[j].pt)
    
    m1=np.array(matches1).reshape(-1,1,2)
    m2=np.array(matches2).reshape(-1,1,2)
    retval,_ = cv2.findHomography(m1, m2, cv2.RANSAC, 5)
    
    #images width and height
    h1,w1,_=img1.shape
    h2,w2,_=img2.shape

    #print(len(matches1))
    tmpwp1 = np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2)
    wp2 = np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)
    wp1= cv2.perspectiveTransform(tmpwp1, retval)
    w = np.concatenate((wp1,wp2), axis=0)

    
    (l, t),(r, b) = np.int32(w.min(axis=0).ravel()), np.int32(w.max(axis=0).ravel())
    translation = np.array([[1.0, 0.0, -l], [0.0, 1.0, -t], [0.0, 0.0, 1.0]])
    warp_img1 = cv2.warpPerspective(img1, translation.dot(retval), (r-l, b-t))
    warp_img2 = cv2.warpPerspective(img2, translation,(r-l, b-t))
    warp_img1[-t:h2-t,-l:w2-l] = img2

    
   
    return warp_img1
   
def stitch(imgmark, N=4, savepath=''): #For bonus: change your input(N=*) here as default if the number of your input pictures is not 4.
    "The output image should be saved in the savepath."
    "The intermediate overlap relation should be returned as NxN a one-hot(only contains 0 or 1) array."
    "Do NOT modify the code provided."
    imgpath = [f'./images/{imgmark}_{n}.png' for n in range(1,N+1)]
    imgs = []
    for ipath in imgpath:
        img = cv2.imread(ipath)
        imgs.append(img)
    "Start you code here"

    new_image=imgs[0]

    for idx in range(1,len(imgs)):
       new_image = stitch_background(new_image,imgs[idx])
       
    cv2.imwrite(savepath, new_image)
    overlap_arr=np.zeros((N,N))
    for idx1, img1 in enumerate(imgs):
        for idx2, img2 in enumerate(imgs):
            olap=Overlap(img1,img2)
            if olap>20:
                overlap_arr[idx1][idx2]=1

    return overlap_arr
if __name__ == "__main__":
    #task2
    overlap_arr = stitch('t2', N=4, savepath='task2.png')
    with open('t2_overlap.txt', 'w') as outfile:
        json.dump(overlap_arr.tolist(), outfile)

    #bonus
    overlap_arr2 = stitch('t3',N=3, savepath='task3.png')
    with open('t3_overlap.txt', 'w') as outfile:
        json.dump(overlap_arr2.tolist(), outfile)
