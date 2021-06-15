import face_recognition
import os
import pickle
from os import path

faceDB=[]

def readEncoding(pathfn):
    print (pathfn)
    with open (pathfn, 'rb') as fp:
        itemlist = pickle.load(fp)
    return itemlist


def findFaceDistances(tfaces,facedist_threshold=0.65):
    faceDistances = []

    for tFace in tfaces['face_encodings']:
        for dbFace in faceDB:
            if tfaces['filehash'] != dbFace['filehash']:
                if (tfaces['numUniquefaces']>0 and dbFace['numUniquefaces']>0):
                    face_distances = face_recognition.face_distance(dbFace['face_encodings'],tFace)
                    for facedist in face_distances:
                        if facedist < facedist_threshold:
                            faceDistances.append([dbFace['fn'],facedist])

    sorted_faceDistances = sorted (faceDistances,key=lambda tup: tup[1])
    return sorted_faceDistances


def buildfaceDB(sfacelist):
    faceDB.clear()
    for encfile in sfacelist:
        dface=readEncoding(encfile)
        faceDB.append(dface)

def dosearch(tface):
    results = []

    for fm in tface:
        dface= readEncoding(fm)
        r=findFaceDistances(dface)
        #fmn=os.path.basename(fm).split('.')[0]+'.jpg'  ##--update needed based on extension
         #chop off the first part of the path
        results.append([dface['fn'],r])
    
    print (r)
    return results



