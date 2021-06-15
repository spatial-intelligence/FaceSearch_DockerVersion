import face_recognition
import os
import sys
import pickle
import imutils
import cv2
from sklearn.cluster import DBSCAN
import facesearch as fs
import numpy as np
import json

def writeEncodingFile(img,mode):
    print ('checking:',img)
    fn_withoutext = os.path.splitext(img)[0]

    configfile = os.path.join(sys.path[0])+'/config.txt'
    with open(configfile,"r") as infile:
        data=json.load(infile)
    print (data[0])

    scansmallfaces= data[0]['smallfacescan']

    if mode=='hog':

        if not os.path.isfile(fn_withoutext+'.fe_hog'):
            image,facelocs,face_encodings,hashcode,numUniqueFaces=getFaceEncodings(img,2,'hog',10,0)
            facedetails = {}
            facedetails['fn']=image
            facedetails['face_locations']=facelocs
            facedetails['face_encodings']=face_encodings
            facedetails['filehash']=hashcode
            facedetails['numUniquefaces']=numUniqueFaces
            writeEncoding(facedetails,fn_withoutext+'.fe_hog')

    if mode =='cnn':
        if not os.path.isfile(fn_withoutext+'.fe_cnn'):
            if scansmallfaces == 'on':
                smallfacescan = 1
                print ('Config file:  small faces scan set ON')
            else:
                smallfacescan = 0
            
            image,facelocs,face_encodings,hashcode,numUniqueFaces=getFaceEncodings(img,2,'cnn',3,smallfacescan)
            facedetails = {}
            facedetails['fn']=image
            facedetails['face_locations']=facelocs
            facedetails['face_encodings']=face_encodings
            facedetails['filehash']=hashcode
            facedetails['numUniquefaces']=numUniqueFaces
            writeEncoding(facedetails,fn_withoutext+'.fe_cnn')

def drawbox(fn,img,facelocations):
    print ('facebox:'+ fn)

    for (top, right, bottom, left) in facelocations:
        cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)
    
    cv2.imwrite(fn, img)

def loadimg (fnimg):
    return face_recognition.load_image_file(fnimg)


def imagehash(img,hashSize=8):
	resized = cv2.resize(img, (hashSize + 1, hashSize))
	diff = resized[:, 1:] > resized[:, :-1]
	return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])


def writeEncoding(enc,pathfn):
    with open(pathfn, 'wb') as fp:
        pickle.dump(enc, fp)


def getFaceEncodings(image,times_upsample, method,jitters=10,smallfacesscan=1):
    
    fr_image= face_recognition.load_image_file(image)
    numUniqueFaces=0
    hashcode = imagehash(fr_image)

    
    if method=='cnn':  ##CNN uses a lot of RAM - first resize to 1000px with 2x upsample to get detail at range of scales -  chop into  sections before detection due to keeping RAM limit low
        (h, w) = fr_image.shape[:2]
        
        max=1000 #maximum size to resize image
        m=int(max/2)

        if h>w:
            fr_image=imutils.resize(fr_image,height=max)
            fr_image1k=imutils.resize(fr_image,height=500) #an overview version of entire image
            print ('resize-h')
            height, width = fr_image.shape[0:2]
            w=int(width/2)
            fr_image_a = fr_image[0:m, 0:w]
            fr_image_b = fr_image[0:m, w:width]
            fr_image_c = fr_image[m:max,0:w]
            fr_image_d = fr_image[m:max,w:width]

        else:
            fr_image=imutils.resize(fr_image,width=max)
            fr_image1k=imutils.resize(fr_image,width=500) #an overview version of entire image
            print ('resize-w')
            height, width = fr_image.shape[0:2]
            h=int(height/2)
            fr_image_a = fr_image[0:h, 0:m]
            fr_image_b = fr_image[0:h, m:max]
            fr_image_c = fr_image[h:height,0:m]
            fr_image_d = fr_image[h:height,m:max]

        print ('cnn: locating faces')

        #Always run with overview image
        print ('checking 1k px overview')
        face_locations = face_recognition.face_locations(fr_image1k, number_of_times_to_upsample=times_upsample, model="cnn")
        #drawbox(image.split('.')[0]+'_facebox_1k.'+image.split('.')[1],fr_image1k,face_locations)   #save image with boxes

        face_encodings= face_recognition.face_encodings(fr_image1k, face_locations)
        

        if smallfacesscan==1:
            print ('check image TL box')
            face_locations_a = face_recognition.face_locations(fr_image_a, number_of_times_to_upsample=times_upsample, model="cnn")
            #drawbox(image.split('.')[0]+'_facebox_a.'+image.split('.')[1],fr_image_a,face_locations_a)   #save image with boxes

            print ('check image TR box')
            face_locations_b = face_recognition.face_locations(fr_image_b, number_of_times_to_upsample=times_upsample, model="cnn")
            #drawbox(image.split('.')[0]+'_facebox_b.'+image.split('.')[1],fr_image_b,face_locations_b)   #save image with boxes

            print ('check image LL box')
            face_locations_c = face_recognition.face_locations(fr_image_c, number_of_times_to_upsample=times_upsample, model="cnn")
            #drawbox(image.split('.')[0]+'_facebox_c.'+image.split('.')[1],fr_image_c,face_locations_c)   #save image with boxes

            print ('check image LR box')
            face_locations_d = face_recognition.face_locations(fr_image_d, number_of_times_to_upsample=times_upsample, model="cnn")
            #drawbox(image.split('.')[0]+'_facebox_d.'+image.split('.')[1],fr_image_d,face_locations_d)   #save image with boxes

            face_encodings_a = face_recognition.face_encodings(fr_image_a, face_locations_a) 
            face_encodings_b = face_recognition.face_encodings(fr_image_b, face_locations_b) 
            face_encodings_c = face_recognition.face_encodings(fr_image_c, face_locations_c)
            face_encodings_d = face_recognition.face_encodings(fr_image_d, face_locations_d) 
        
            face_encodings=face_encodings+face_encodings_a+face_encodings_b+face_encodings_c+face_encodings_d

        facelocs = face_locations 
        
        #get unique list of face encodings from multiple searches ---- TO DO work in progress for drawing boxes
        #print("[INFO] clustering...")
        #clt = DBSCAN(metric="euclidean", n_jobs=2)
        #clt.fit(face_encodings)
        #labelIDs = np.unique(clt.labels_)
        #determine the total number of unique faces found in the dataset
        #numUniqueFaces = len(np.where(labelIDs > -1)[0])
        numUniqueFaces = len (face_encodings)

    else: 
        #HOG method for speed with jitters for increased accuracy (random image warps)

        (h, w) = fr_image.shape[:2]
        max=2000 #maximum size to resize image
        m=int(max/2)

        if h>w:
            fr_image=imutils.resize(fr_image,height=max)
        else:
            fr_image=imutils.resize(fr_image,width=max)

        facelocs = face_recognition.face_locations(fr_image, number_of_times_to_upsample=times_upsample, model='hog')    
        face_encodings= face_recognition.face_encodings(fr_image,facelocs,num_jitters=jitters)
        numUniqueFaces = len (face_encodings)
        #drawbox(image.split('.')[0]+'_facebox_hog.'+image.split('.')[1],fr_image,facelocs)   #save image with boxes
    

    print (len(face_encodings))
    print (hashcode)
    print ('=====')

    return image,facelocs,face_encodings,hashcode,numUniqueFaces
