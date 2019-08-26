import argparse
import os
import pickle
import numpy as np
import cv2

from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

from extract import calc_embs


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
    folder_model = args.model
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
        with open(folder_model+"/svc_model.pickle","rb") as f:
            clf = pickle.load(f)

        with open(folder_model+"/encoder.pickle","rb") as f:
            le = pickle.load(f)

        while True:
            print("Enter your image file...")
            filename = input()
            if filename == "":
                print("Do you want to quit(y/n)?")
                msg = input()
                if msg == "y":
                    print("Byebye , see you again.")
                    break
                elif msg == "n":
                    print("Enter your image file...")
                    filename = input()
                    pass

            embs,boxs,imgs = calc_embs([filename])

            embs = np.concatenate(embs)

            labels = le.inverse_transform(clf.predict([embs]))

            print(labels)

            copy = imgs[0].copy()
            x,y,w,h = boxs[0]
            cv2.rectangle(copy, (x,y), (x++w,y+h), (255,0,0),2)
            cv2.putText(copy,labels[0],(x,y),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)
            cv2.imshow("",copy)
            cv2.waitKey(0)

            cv2.imwrite("output.jpg",copy)
            labels = ",".join(labels)

            with open("labels.txt","w") as of:
                of.write(labels)

if __name__ == "__main__":
    train()

