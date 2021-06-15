#to run  python flaskappname.py     if you have app.run () details in the __main__
# or set env variable:     export FLASK_APP=flaskappname.py       and then     flask run

#sync to server

from flask import Flask, render_template,url_for, Response, request,send_from_directory
import glob
import os
import sys
from PIL import Image, ImageDraw
import encode_images as enc
import findtarget as findtarget
import time
from pandas import DataFrame
import numpy as np
import encode_images as ei
import findtarget as ft
import json

app = Flask(__name__)

path = '/data/osr4rights/'

app.config['CUSTOM_STATIC_PATH']= path+'/'  #gives access to the DATA folder mapped to the local host drive 

search_imgpath = path + 'search/'
target_imgpath = path + 'target/'

# Custom static data
@app.route('/cdn/<path:filename>')
def custom_static(filename):
    return send_from_directory(app.config['CUSTOM_STATIC_PATH'], filename)


def checkimages_against_hashes():
    filechanges = []
    filecount=0

    try:

        for img in glob.glob(search_imgpath+'*.jpg')+ glob.glob(search_imgpath+'*.png')+glob.glob(search_imgpath+'*.JPG')+ glob.glob(search_imgpath+'*.PNG'):

            hogfiles = (glob.glob(search_imgpath+'*.fe_hog'))
            cnnfiles = (glob.glob(search_imgpath+'*.fe_cnn'))

            hfiles = [w.replace('.fe_hog', '') for w in hogfiles]
            cfiles = [w.replace('.fe_cnn', '') for w in cnnfiles]

            if (img.split('.')[0] in hfiles):   #-- need to scan just first part of filename against first part in the list 
                fnhash= ft.readEncoding(img.split('.')[0]+'.fe_hog')['filehash']
                filecount += 1
                hashimg=ei.imagehash(ei.loadimg(img))
                if hashimg != fnhash:
                    filechanges.append(img)

            if (img.split('.')[0] in cfiles):   #-- need to scan just first part of filename against first part in the list 
                fnhash= ft.readEncoding(img.split('.')[0]+'fe_cnn')['filehash']
                filecount += 1
                hashimg=ei.imagehash(ei.loadimg(img))
                if hashimg != fnhash:
                    filechanges.append(img)
    except:
        print ('Error with a face encoding file')

    return filechanges,filecount
    


def filescan():
    hogfacesfound = []
    for img in (glob.glob(search_imgpath+'*.fe_hog')):

            facecount= ft.readEncoding(img)['numUniquefaces']
            hogfacesfound.append([facecount,img])

    cnnfacesfound = []

    for img in (glob.glob(search_imgpath+'*.fe_cnn')):

            facecount= ft.readEncoding(img)['numUniquefaces']
            cnnfacesfound.append([facecount,img])

    return hogfacesfound,cnnfacesfound



def searchfiles():
    fehog = len (glob.glob(search_imgpath+'*.fe_hog'))
    fecnn = len (glob.glob(search_imgpath+'*.fe_cnn'))

    return glob.glob(search_imgpath+'*.jpg')+ glob.glob(search_imgpath+'*.png')+glob.glob(search_imgpath+'*.JPG')+ glob.glob(search_imgpath+'*.PNG'),fehog,fecnn

def searchfiles_encoding_hog():
    fehog = (glob.glob(search_imgpath+'*.fe_hog'))
    return fehog

def searchfiles_encoding_cnn():
    fehog = (glob.glob(search_imgpath+'*.fe_cnn'))
    return fehog

def targetfiles():
    fehog = len (glob.glob(target_imgpath+'*.fe_hog'))
    fecnn = len (glob.glob(target_imgpath+'*.fe_cnn'))
    return glob.glob(target_imgpath+'*.jpg')+ glob.glob(target_imgpath+'*.png')+glob.glob(target_imgpath+'*.JPG')+ glob.glob(target_imgpath+'*.PNG'),fehog,fecnn

def targetfiles_encoding_hog():
    fehog = (glob.glob(target_imgpath+'*.fe_hog'))
    return fehog

def targetfiles_encoding_cnn():
    fehog = (glob.glob(target_imgpath+'*.fe_cnn'))
    return fehog


def imagetopdf():
    files = scanfiles()
    img_notes =img = Image.new('RGB', (800,100), color = 'white')
    d = ImageDraw.Draw(img_notes)
    d.text((10,10), "OSR4Rights: FaceSearch Report", fill=(10,10,10))

    image1 = Image.open(files[0])
    image2 = Image.open(files[1])

    im1 = image1.convert('RGB')
    im2 = image2.convert('RGB')

    imagelist = [im1,im2]
    date_time = now.strftime("%m_%d_%Y_%H_%M_%S")
    img_notes.save(path+'report_'+date_time+'.pdf',save_all=True, append_images=imagelist)


@app.route("/home")
@app.route("/")
def home():
    return render_template('home.html',title='home')


@app.route("/about")
def about():
    return render_template('about.html',title='about')


@app.route('/procimageshog')
def procimageshog():

    def generate():
        sfiles,sfehog,sfecnn  = searchfiles()
        tfiles,tfehog,tfecnn  = targetfiles()
        filecount=len(sfiles) + len(tfiles)
        x=0.0

        #calc search images
        for img in sfiles:
            print (img)
            enc.writeEncodingFile(img,'hog')
            yield "data:" + str(int(x)) + "\n\n"
            x = x + (1/filecount*100)
        
        #now for target images
        for img in tfiles:
            print (img)
            enc.writeEncodingFile(img,'hog')
            yield "data:" + str(int(x)) + "\n\n"
            x = x + (1/filecount*100)

        x=100 # finish exactly on 100 to trigger URL redirect from progressbar page
        yield "data:" + str(int(x)) + "\n\n"

    return Response(generate(), mimetype= 'text/event-stream')



@app.route('/procimagescnn')
def procimagescnn():

    def generate():
        sfiles,sfehog,sfecnn = searchfiles()
        tfiles,tfehog,tfecnn  = targetfiles()
        filecount=len(sfiles) + len(tfiles)
        x=1.0

        #calc search images
        for img in sfiles:
            print (img)
            enc.writeEncodingFile(img,'cnn')
            yield "data:" + str(int(x)) + "\n\n"
            x = x + (1/filecount*100)
        
        #now for target images
        for img in tfiles:
            print (img)
            enc.writeEncodingFile(img,'cnn')
            yield "data:" + str(int(x)) + "\n\n"
            x = x + (1/filecount*100)

        x=100 # finish exactly on 100 to trigger URL redirect from progressbar page
        yield "data:" + str(int(x)) + "\n\n"

    return Response(generate(), mimetype= 'text/event-stream')



@app.route("/process1", methods=['GET', 'POST'])
def process1():
    sfiles,sfehog,sfecnn = searchfiles()  #run hashcheck on the images against any previous encodings (check files tab)
    tfiles,tfehog,tfecnn = targetfiles()
    return render_template('process1.html',sdata=sfiles[0:4],sitemtotal=len(sfiles),sfehogcount=sfehog,sfecnncount=sfecnn,tdata=tfiles[0:4],titemtotal=len(tfiles),tfehogcount=tfehog,tfecnncount=tfecnn,title='home')


@app.route("/process2hog")
def process2hog():
    files,fehog,fecnn = searchfiles()
    filecount=len(files)
    return render_template('process2hog.html')

@app.route("/process2cnn")
def process2cnn():
    files,fehog,fecnn = searchfiles()
    filecount=len(files)
    return render_template('process2cnn.html')


@app.route("/process2details")
def process2details():
    hogfacesfound,cnnfacesfound = filescan()
    hogfacesfoundsorted=sorted(hogfacesfound, key = lambda x: (x[0],x[1]))
    cnnfacesfoundsorted=sorted(cnnfacesfound, key = lambda x: (x[0],x[0]))
    return render_template('process2details.html',hogfound=hogfacesfoundsorted,cnnfound =cnnfacesfoundsorted )


@app.route("/process3", methods=['GET', 'POST'])
def process3():
    sfiles,sfehog,sfecnn = searchfiles()
    tfiles,tfehog,tfecnn = targetfiles()

    if request.method == 'POST':
        if request.form.get('v') == 'HOG':
            # pass
            print("hog")
            results = searchfortarget('HOG')
            return render_template("results.html",res= results)

        elif  request.form.get('v') == 'CNN':
            # pass # do something else
             print("cnn")
             results = searchfortarget('CNN')

             return render_template("results.html",res= results)

    elif request.method == 'GET':
        # return render_template("index.html")
        print ('.')

    return render_template("process3.html",shog=sfehog,scnn=sfecnn,thog=tfehog,tcnn=tfecnn)


@app.route("/setconfig", methods=['GET', 'POST'])
def setconfig():

    sf = 'on'

    print ('small faces toggle')

    if request.method == 'POST':
        if request.form.get('v') == 'smallfaces_on':
            # pass
            print("smallfaces ON")
            sf ='on'
            

        elif  request.form.get('v') == 'smallfaces_off':
            # pass # do something else
             print("smallface OFF")
             sf='off'


    configfile = os.path.join(sys.path[0])+'/config.txt'
    with open(configfile,"w") as outfile:
        data=[]
        data.append ({'smallfacescan':sf})
        json.dump(data,outfile)

    return render_template('smallfaces.html', sftoggle = sf)
    
#Delete the ENCODING files in the SEARCH folder
@app.route("/deleteencodings", methods=['GET', 'POST'])
def deleteencodings():

    if request.method == 'POST':
        if request.form.get('v') == 'delete hog':
            for fn in searchfiles_encoding_hog():
                os.remove (fn)
            for fn in targetfiles_encoding_hog():
                os.remove (fn)

        if request.form.get('v') == 'delete cnn':
            for fn in searchfiles_encoding_cnn():
                os.remove (fn)
            for fn in targetfiles_encoding_cnn():
                os.remove (fn)

    sfiles,sfehog,sfecnn = searchfiles()  #run hashcheck on the images against any previous encodings (check files tab)
    tfiles,tfehog,tfecnn = targetfiles()
    return render_template('process1.html',sdata=sfiles[0:4],sitemtotal=len(sfiles),sfehogcount=sfehog,sfecnncount=sfecnn,tdata=tfiles[0:4],titemtotal=len(tfiles),tfehogcount=tfehog,tfecnncount=tfecnn,title='home')

    





def searchfortarget(mode):

    if mode=='HOG':
        #load all the searchfile face encodings
        senc = searchfiles_encoding_hog()
        findtarget.buildfaceDB(senc)
        #load all the target face encodings and do comparison
        tenc = targetfiles_encoding_hog()
        res = findtarget.dosearch(tenc)

        #unique list of matches (images) - showing best score match per image
        df = DataFrame (res,columns=['fn','diff'])
       
 
        uniqueresults = []
        for index, row in df.iterrows():
            r = (row['diff'])
            minscores={}
            for i in r:
                f=i[0].replace(path,'')
                if f in minscores and minscores[f] > i[1]:
                    minscores[f]=i[1]
                if f not in minscores:
                    minscores[f]=i[1]
            uniqueresults.append([row['fn'].replace(path,''), minscores.items() ])

        #uniqueresults=df.values.tolist()

        return uniqueresults

    elif mode=='CNN':
        #load all the searchfile face encodings
        senc = searchfiles_encoding_cnn()
        findtarget.buildfaceDB(senc)

        #load all the target face encodings and do comparison
        tenc = targetfiles_encoding_cnn()
        res = findtarget.dosearch(tenc)

         #unique list of matches (images) - showing best score match per image
        df = DataFrame (res,columns=['fn','diff'])
 
        uniqueresults = []
        for index, row in df.iterrows():
            r = (row['diff'])
            minscores={}
            for i in r:
                f=i[0].replace(path,'')
                if f in minscores and minscores[f] > i[1]:
                    minscores[f]=i[1]
                if f not in minscores:
                    minscores[f]=i[1]
            uniqueresults.append([row['fn'].replace(path,''), minscores.items() ])
        
        #uniqueresults=df.values.tolist()
            
        return uniqueresults



@app.route("/results", methods=['GET', 'POST'])
def results():

    return render_template('results.html',itemtotal=filecount,title='results')



@app.route("/checkfilehash", methods=['GET', 'POST'])
def checkfilehash():
    #check if file has fe_* that matches the image hash
    filechanges=[]

    print ('running file hash check')
    filechanges,filecount = checkimages_against_hashes()
    return render_template('checkfilehash.html',shashcheck=filechanges,fc = filecount)



@app.route("/img/<fn>")
def check(fn): 
    return f'filename:,{fn}'


if __name__ == "__main__":
    #app.run(debug=True)
    print ('Running Server')

    app.run(host='0.0.0.0',port=5050, debug=True)


