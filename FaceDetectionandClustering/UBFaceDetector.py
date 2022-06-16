'''
All of your implementation should be in this file.
'''
'''
This is the only .py file you need to submit. 
'''
'''
    Please do not use cv2.imwrite() and cv2.imshow() in this function.
    If you want to show an image for debugging, please use show_image() function in helper.py.
    Please do not save any intermediate files in your final submission.
'''
from unittest import result
from helper import show_image

import cv2
import numpy as np
import os
import sys

import face_recognition
from sklearn.cluster import KMeans,DBSCAN
import matplotlib.pyplot as plt
import matplotlib.gridspec as grid
'''
Please do NOT add any imports. The allowed libraries are already imported for you.
'''


def detect_faces(input_path: str) -> dict:
    result_list = []
    '''
    Your implementation.
    '''
    for image in os.listdir(input_path):
        img_path=os.path.join(input_path,image)
        img=cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
        faces=cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
        found_face_loc=faces.detectMultiScale(img,scaleFactor=1.12,minNeighbors=4)
        if len(found_face_loc)>0:
            for val in found_face_loc:
                val= [int(i) for i in val]
                dict={"iname":image,"bbox":val}
                result_list.append(dict)
    #print(len(result_list))
    return result_list


'''
K: number of clusters
'''
def cluster_faces(input_path: str, K: int) -> dict:
    result_list = []
    '''
    Your implementation.
    '''
    
    labels=[]
    temp=detect_faces(input_path)
    #print(len(temp))
    count=0
    data=[]
    for val in temp:
        x,y,w,h=val['bbox']
        img_path=os.path.join(input_path,val['iname'])
        img=face_recognition.face_encodings(cv2.imread(img_path), [(y,x+w,y+h,x)])
        labels.append(val['iname'])
        if(count==0):
            data=img[0]
        else:
            data=np.vstack((data,img[0]))
        count+=1
    #kmeans = KMeans(n_clusters=int(K),init='k-means++')
    #kmeans.fit(data) 
    #clabels=kmeans.labels_
    #print(len(clabels))
    db=DBSCAN(metric="euclidean").fit(data)
    clabels=db.labels_
    
    for i in range(int(K)):
        result_list.append({"cluster_no":i,"elements":[]})
    for i,ele in enumerate(clabels):
        for j,val in enumerate(result_list):
            if(val['cluster_no']==ele):
                result_list[j]['elements'].append(labels[i])
    show(input_path,result_list)
    
    return result_list


'''
If you want to write your implementation in multiple functions, you can write them here. 
But remember the above 2 functions are the only functions that will be called by FaceCluster.py and FaceDetector.py.
'''

"""
Your implementation of other functions (if needed).
"""
def show(input_path,result):
    for c in result:
        fig = plt.figure(figsize=(10, 7))
        images=[]
        img_list=c['elements']
        
        mod=int(len(img_list)%5)
        if(mod==0):
            x=int(len(img_list)/5)
        else:
            x=int(len(img_list)/5)+1
        y=5
       
        fig=plt.figure(figsize=(5,5))
        g=grid.GridSpec(5,5)
        g.update(wspace=0,hspace=0)
        fig.suptitle("Cluster "+str(c['cluster_no']))
        for i in np.arange(0, x*y):
            ax=plt.subplot(g[i])
            if(i<=len(img_list)-1):
                img_path=os.path.join(input_path,img_list[i])
                img=cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB )
                img=cv2.resize(img, (100,100))
                ax.imshow(img)
    
            else:
                img=np.ones((100,100,3))
                #print(img)
                ax.imshow(img)
            ax.axis('off')
        fname=str(c['cluster_no'])+".png"
        plt.savefig(fname)
        

