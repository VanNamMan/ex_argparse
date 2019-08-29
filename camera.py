import cv2
import numpy as np
import time,os,json

"""
key :
    q : exit 
    s : save iamge
    o : on all led
    f : off all led
    d : open camera setting dialog
"""
def mkdir(folder):
    if not os.path.exists(folder):
        os.mkdir(folder)

print(__doc__)

folder = os.path.join(os.getcwd(),"images")
mkdir(folder)
 
cap = cv2.VideoCapture(0)
print("camera 0 : ",cap.isOpened())

print("Size(%d,%d)"%(cap.get(cv2.CAP_PROP_FRAME_WIDTH),cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

window = cv2.namedWindow("Dino Camera",cv2.WINDOW_FREERATIO)
folder_name = folder

def main():
    while cap.isOpened():
        ret,img = cap.read()
        if ret :
            cv2.imshow("Dino Camera",img)
            k = cv2.waitKey(20)
            if k == ord("q"):
                break
            elif k == ord("f"):
                label = input("Enter label : \n")
                folder_name = os.path.join(folder,label)
                print(folder_name)
                mkdir(folder_name)
            elif k==ord("s"):
                filename = os.path.join(folder_name,time.strftime("%d%m%y_%H%M%S.jpg"))
                if cv2.imwrite(filename,img):
                    print(filename)
            elif k == ord("o"):
                os.system("DN_DS_Ctrl.exe LED on 1 -CAM0")
            elif k == ord("f"):
                os.system("DN_DS_Ctrl.exe LED off -CAM0")
            elif k == ord("d"):
                cap.set(cv2.CAP_PROP_SETTINGS, 1)
        else:
            print(ret)
            print("Disconnect.")
            break          
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()