import cv2
import os


videoname = "videoname"
foldername = "foldername"
cam = cv2.VideoCapture("Video-Bounding-Box-Labelling-Tool-master/data/"+videoname+".avi")

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

createFolder("./"+foldername+"/")

n=0
while(True):

    ret, frame = cam.read()
    frame = cv2.resize(frame,(854,480))
    if ret:
        name = "Video-Bounding-Box-Labelling-Tool-master/data/"+foldername+'/'+ str(n).zfill(5) + ".jpg"
        print(name)
        cv2.imwrite(name,frame)
        n=n+1
    else:
        break  

cam.release()
cv2.destroyAllWindows()  
