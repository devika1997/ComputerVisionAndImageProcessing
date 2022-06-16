#Only add your code inside the function (including newly improted packages)
# You can design a new function and call the new function in the given functions. 
# Not following the project guidelines will result in a 10% reduction in grades

import cv2
import numpy as np
import matplotlib.pyplot as plt

def stitch_background(img1, img2, savepath=''):
    "The output image should be saved in the savepath."
    "Do NOT modify the code provided."
    matches1=[]
    matches2=[]
    keypoints1, descriptors1 = cv2.SIFT_create(200).detectAndCompute(img1,None)
    keypoints2, descriptors2 = cv2.SIFT_create(200).detectAndCompute(img2,None)
    for i in range(len(descriptors1)):
        for j in range(len(descriptors2)):
           dist=np.linalg.norm(descriptors1[i]-descriptors2[j])/len(descriptors1)
           if(dist<1.1):
                matches1.append(keypoints1[i].pt)
                matches2.append(keypoints2[j].pt)
    m1=np.asarray(matches1).reshape(-1,1,2)
    m2=np.asarray(matches2).reshape(-1,1,2)
    retval,_ = cv2.findHomography(m1, m2, cv2.RANSAC, 5)

    #images width and height
    h1,w1,_=img1.shape
    h2,w2,_=img2.shape

    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    i1=cv2.threshold(img1_gray, 1,255, cv2.THRESH_BINARY)[1]
    i2=cv2.threshold(img2_gray, 1,255, cv2.THRESH_BINARY)[1]

    tmpwp1 = np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2)
    wp2 = np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)
    wp1= cv2.perspectiveTransform(tmpwp1, retval)
    w = np.concatenate((wp1,wp2), axis=0)

    (l, t),(r, b) = np.int32(w.min(axis=0).ravel()), np.int32(w.max(axis=0).ravel())
    translation = np.array([[1.0, 0.0, -l], [0.0, 1.0, -t], [0.0, 0.0, 1.0]])
    warp_img1 = cv2.warpPerspective(img1, translation.dot(retval), (r-l, b-t))
    warp_img2 = cv2.warpPerspective(img2, translation.dot(retval),(r-l, b-t))
    intersection = cv2.subtract(warp_img1,warp_img2)
    warp_img1[-t:h2-t,-l:w2-l] = img2
    _, mask = cv2.threshold(cv2.cvtColor(intersection, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY)
    
    temp = cv2.warpPerspective(img1, translation.dot(retval), (r-l, b-t))
    for idx,val in np.ndenumerate(mask):
       if val>0:
           warp_img1[idx[0],idx[1]]=temp[idx[0],idx[1]]
    
    cv2.imwrite(savepath, warp_img1)
    return
if __name__ == "__main__":
    img1 = cv2.imread('./images/t1_1.png')
    img2 = cv2.imread('./images/t1_2.png')
    savepath = 'task1.png'
    stitch_background(img1, img2, savepath=savepath)

