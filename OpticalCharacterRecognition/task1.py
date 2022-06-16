"""
Character Detection

The goal of this task is to implement an optical character recognition system consisting of Enrollment, Detection and Recognition sub tasks

Please complete all the functions that are labelled with '# TODO'. When implementing the functions,
comment the lines 'raise NotImplementedError' instead of deleting them.

Do NOT modify the code provided.
Please follow the guidelines mentioned in the project1.pdf
Do NOT import any library (function, module, etc.).
"""
# OpenCV 4.5.4

import argparse
import json
import os
import glob
import cv2
import numpy as np


def read_image(img_path, show=False):
    """Reads an image into memory as a grayscale array.
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if show:
        show_image(img)

    return img

def show_image(img, delay=1000):
    """Shows an image.
    """
    cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('image', img)
    cv2.waitKey(delay)
    cv2.destroyAllWindows()

def parse_args():
    parser = argparse.ArgumentParser(description="cse 473/573 project 1.")
    parser.add_argument(
        "--test_img", type=str, default="./data/test_img.jpg",
        help="path to the image used for character detection (do not change this arg)")
    parser.add_argument(
        "--character_folder_path", type=str, default="./data/characters",
        help="path to the characters folder")
    parser.add_argument(
        "--result_saving_directory", dest="rs_directory", type=str, default="./",
        help="directory to which results are saved (do not change this arg)")
    args = parser.parse_args()
    return args

def ocr(test_img, characters):
    """Step 1 : Enroll a set of characters. Also, you may store features in an intermediate file.
       Step 2 : Use connected component labeling to detect various characters in an test_img.
       Step 3 : Taking each of the character detected from previous step,
         and your features for each of the enrolled characters, you are required to a recognition or matching.

    Args:
        test_img : image that contains character to be detected.
        characters_list: list of characters along with name for each character.

    Returns:
    a nested list, where each element is a dictionary with {"bbox" : (x(int), y (int), w (int), h (int)), "name" : (string)},
        x: row that the character appears (starts from 0).
        y: column that the character appears (starts from 0).
        w: width of the detected character.
        h: height of the detected character.
        name: name of character provided or "UNKNOWN".
        Note : the order of detected characters should follow english text reading pattern, i.e.,
            list should start from top left, then move from left to right. After finishing the first line, go to the next line and continue.
        
    """
    # TODO Add your code here. Do not modify the return and input arguments
    characterlist=enrollment(characters)

    pos_dict,features=detection(test_img)
    
    results=recognition(characterlist,pos_dict,features)

    return results
    #raise NotImplementedError

def enrollment(characters):
    """ Args:
        You are free to decide the input arguments.
    Returns:
    You are free to decide the return.
    """
    # TODO: Step 1 : Your Enrollment code should go here.
    mean_char=[]
    for i in range(len(characters)):
        mean_char.append(np.mean(characters[i][1]))
    
    for i in range(len(characters)):
        for index,val in np.ndenumerate(characters[i][1]):
            if val>min(mean_char):
                characters[i][1][index[0]][index[1]] = 0
            else:
                characters[i][1][index[0]][index[1]] = 255

    #canny for extracting features

    for i in range(len(characters)):
        descriptors = cv2.Canny(characters[i][1],100,200)
        idx = np.argwhere(np.all(descriptors[..., :] == 0, axis=0))
        descriptors = np.delete(descriptors, idx, axis=1)
        descriptors = descriptors[~np.all(descriptors == 0, axis=1)]
        pad_arr = np.pad(descriptors, (1,),'constant', constant_values=0) 
        characters[i][1]=pad_arr
        cv2.imwrite(f"./features/feature{i}.jpeg",characters[i][1])
    return characters
    #raise NotImplementedError
    
def detection(test_img):
    """ 
    Use connected component labeling to detect various characters in an test_img.
    Args:
        You are free to decide the input arguments.
    Returns:
    You are free to decide the return.
    """
    # TODO: Step 2 : Your Detection code should go here.

    for index,val in np.ndenumerate(test_img):
        if val>170:
            test_img[index] = 0
        else:
            test_img[index] = 255
    
    #Connected Component Labelling
    labels=np.zeros(test_img.shape)
    c_label=1

    #First pass
    for idx,val in np.ndenumerate(test_img):
        y=idx[0]
        x=idx[1]
        if(val!=0):
            left_label=labels[y,x-1]
            above_label=labels[y-1,x]
            left_pixel=test_img[y,x-1]
            above_pixel=test_img[y-1,x]
            if(left_pixel==0 and above_pixel==0):
                labels[y,x]=c_label
                c_label+=1
            else:
                if(above_pixel==val):
                    labels[y][x]=above_label
                elif(left_pixel==val):
                    labels[y][x]=left_label
                
    
    #Second Pass
    for idx,val in np.ndenumerate(labels):
        y=idx[0]
        x=idx[1]
        if(val!=0):
            if((test_img[y,x]==test_img[y,x-1] )and (val!=labels[y,x-1]) and labels[y,x-1]!=0):
                labels[labels==val]=labels[y,x-1]
    

    #Retrieving top left corner and bottom right corner for each character
    pos_dict=dict()
    for idx,val in np.ndenumerate(labels):
        y=idx[0]
        x=idx[1]
        if(val!=0):
            if val not in pos_dict:
                pos_dict[int(val)]=[y,x,y,x]  #postion 0-top,1-left,2-bottom,3-right
            else:
                if(y<pos_dict[val][0]):
                    pos_dict[val][0]=y
                if(x<pos_dict[val][1]):
                    pos_dict[val][1]=x
                if(y>pos_dict[val][2]):
                    pos_dict[val][2]=y
                if(x>pos_dict[val][3]):
                    pos_dict[val][3]=x

    #Cropping and Padding
    img_dict=dict()
    for key in pos_dict:
        top=pos_dict[key][0]
        left=pos_dict[key][1]
        bottom=pos_dict[key][2]
        right=pos_dict[key][3]
        img_dict[key]=test_img[top:bottom+1,left:right+1].astype(np.uint8)

    #Padding and Extracting features for each character in test_img
    features=dict()
    for key in img_dict:
        pad_img=np.pad(img_dict[key], (1,),'constant', constant_values=0)
        descriptors = cv2.Canny(pad_img,100,200)
        pad_img1=np.pad(descriptors, (1,),'constant', constant_values=0)
        features[key]=pad_img1

    return pos_dict,features
    #raise NotImplementedError


def recognition(characterlist,pos_dict,testimg_features):
    """ 
    Args:
        You are free to decide the input arguments.
    Returns:
    You are free to decide the return.
    """
    # TODO: Step 3 : Your Recognition code should go here.
    #Normalised Cross Correlation
    bbox_list=[]
    count=0
    for key in testimg_features:
        ncc_s=0
        chari=0
        char='UNKNOWN'
        for i in range(len(characterlist)):
            test_idx=testimg_features[key].shape
            idx=characterlist[i][1].shape
            img_resize = cv2.resize(characterlist[i][1],(test_idx[1],test_idx[0]))
            
            for index,val in np.ndenumerate(img_resize):
                if val>0:
                    img_resize[index] = 255
            
            mean_tf=np.mean(testimg_features[key])
            mean_imgt=np.mean(img_resize)
            ncc=((testimg_features[key]-mean_tf)*(img_resize-mean_imgt))/(np.std(img_resize)*np.std(testimg_features[key]))
            ncc=np.sum(ncc)/(test_idx[1]*test_idx[0])
            
            if(ncc>ncc_s):
                ncc_s=ncc
                chari=i
        
        if(ncc_s>0.37):
            count+=1
            char=characterlist[chari][0]
        
        bbox_list.append({'bbox':[pos_dict[key][1],pos_dict[key][0],pos_dict[key][3]-pos_dict[key][1],pos_dict[key][2]-pos_dict[key][0]],'name':char})
    
    #print(f"count:{count}")
    return bbox_list

    #raise NotImplementedError


def save_results(coordinates, rs_directory):
    """
    Donot modify this code
    """
    results = coordinates
    with open(os.path.join(rs_directory, 'results.json'), "w") as file:
        json.dump(results, file)


def main():
    args = parse_args()
    
    characters = []

    all_character_imgs = glob.glob(args.character_folder_path+ "/*")
    
    for each_character in all_character_imgs :
        character_name = "{}".format(os.path.split(each_character)[-1].split('.')[0])
        characters.append([character_name, read_image(each_character, show=False)])

    test_img = read_image(args.test_img)

    results = ocr(test_img, characters)

    save_results(results, args.rs_directory)


if __name__ == "__main__":
    main()