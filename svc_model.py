import argparse
import os
import pickle
import numpy as np
import cv2

from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

def predict(filename,calc_embs,encoder,clf):
    embs,boxs,imgs = calc_embs([filename])

    if embs is not None:
        copy = imgs[0].copy()
        embs = np.concatenate(embs)

        labels = encoder.inverse_transform(clf.predict([embs]))

        print(labels)
        # x,y,w,h = boxs[0]

        return labels[0],boxs[0],copy
    else:
        copy = cv2.imread(filename,0)
        return None,None,copy

def train():
    parser = argparse.ArgumentParser(description='Classification with SVM.')

    parser.add_argument('-e','--embdding', type=str,help='folder of embdding')
    parser.add_argument('-n','--num', type=int,help='max num embdding for each label')
    parser.add_argument('-model','--model', type=str,help='folder of model SVC')

    parser.add_argument('-m','--mode', type=str,help='mode train or test')

    # parser.add_argument('-i','--image_path', type=str,help='path of image test')

    args = parser.parse_args()

    embdding_path = args.embdding

    max_num_img = args.num
    print("max_num_img : ",max_num_img)

    folder_model = args.model
    if not os.path.exists(folder_model):
        os.mkdir(folder_model)

    mode = args.mode
    # filename = args.image_path

    if mode == "train":

        labels = []
        embs = []
        filenames = os.listdir(embdding_path)

        for f in filenames:
            embs_ = pickle.load(open(os.path.join(embdding_path,f),"rb"))[:max_num_img,:]
            labels.extend([f.split(".")[0]] * len(embs_))
            embs.append(embs_)
        
        embs = np.concatenate(embs)
        print(embs.shape)
        print(labels)
        le = LabelEncoder().fit(labels)
        y = le.transform(labels)
        print(y)
        clf = SVC(kernel='linear', probability=True).fit(embs, y)
        print(clf)

        with open(folder_model+"/svc_model.pickle","wb") as f:
            pickle.dump(clf,f)

        with open(folder_model+"/encoder.pickle","wb") as f:
            pickle.dump(le,f)

    elif mode == "test":
        from extract import calc_embs
        wd = cv2.namedWindow("",cv2.WINDOW_FREERATIO)

        with open(folder_model+"/svc_model.pickle","rb") as f:
            clf = pickle.load(f)

        with open(folder_model+"/encoder.pickle","rb") as f:
            le = pickle.load(f)

        while True:
            filename = ""
            timeout = 100

            filename = input("Enter your image file : ")
            if filename == "":
                msg = input("Do you want to quit(y/n)? : ")
                if msg == "y":
                    print("Byebye , see you again.")
                    break
                elif msg == "n":
                    filename = input("Enter your image file : ")

            if os.path.isfile(filename) and os.path.exists(filename):

                label , box , copy = predict(filename,calc_embs,le,clf)
                if label is not None:
                    x,y,w,h = box
                    cv2.rectangle(copy, (x,y), (x++w,y+h), (255,0,0),2)
                    cv2.putText(copy,label,(x,y),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)
                cv2.imshow(wd,copy)
                cv2.waitKey(0)

                cv2.imwrite("output.jpg",copy)
                # labels = ",".join(labels)

                with open("labels.txt","w") as of:
                    of.write(label)

            elif os.path.isdir(filename) and os.path.exists(filename):
                timeout = input("Enter timeout : ")
                try:
                    timeout = int(timeout)
                except:
                    timeout = 100

                for f in os.listdir(filename):
                    file = os.path.join(filename,f)
                    label , box , copy = predict(file,calc_embs,le,clf)

                    if label is not None:
                        x,y,w,h = box
                        cv2.rectangle(copy, (x,y), (x++w,y+h), (255,0,0),2)
                        cv2.putText(copy,label,(x,y),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)
                    cv2.imshow(wd,copy)
                    cv2.waitKey(timeout)

        cv2.destroyAllWindows()

if __name__ == "__main__":
    train()

