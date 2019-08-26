import argparse
from keras.models import load_model
import os
import pickle
from mtcnn.mtcnn import MTCNN
import cv2
import numpy as np

image_size = 160
model = load_model("model/facenet_keras.h5")
face_mtcnn = MTCNN(steps_threshold=[0.7]*3
                ,scale_factor=0.8,min_face_size=80)


def prewhiten(x):
    if x.ndim == 4:
        axis = (1, 2, 3)
        size = x[0].size
    elif x.ndim == 3:
        axis = (0, 1, 2)
        size = x.size
    else:
        raise ValueError('Dimension should be 3 or 4')

    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True)
    std_adj = np.maximum(std, 1.0/np.sqrt(size))
    y = (x - mean) / std_adj
    return y

def l2_normalize(x, axis=-1, epsilon=1e-10):
    output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
    return output

def get_bounding_boxes(mtcnn_detech):
#     print(mtcnn_detech)
    box = list(map(int,mtcnn_detech["box"]))
    score = mtcnn_detech["confidence"]
    kp = mtcnn_detech["keypoints"]
    points = [kp["left_eye"],kp["right_eye"],kp["nose"],kp["mouth_left"],kp["mouth_right"]]
    return box,score,points

def cropped(img,box,margin=10):

    h0,w0 = img.shape[:2]
    x,y,w,h = box

    x1 = max(0,x-margin//2)
    y1 = max(0,y-margin//2)

    x2 = min(x+w+margin//2,w0)
    y2 = min(y+h+margin//2,h0)

    cropped = img[y1:y2,x1:x2,:]
    aligned = cv2.resize(cropped, (image_size, image_size),interpolation=cv2.INTER_CUBIC)
    return aligned

def load_and_align_images(filepaths, margin):
    aligned_images = []
    boxs = []
    imgs = []
    for filepath in filepaths:
        img = cv2.imread(filepath)

        faces = face_mtcnn.detect_faces(img)
        if len(faces) > 0:
            box,score,points = get_bounding_boxes(faces[0])

            (x, y, w, h) = box
            aligned = cropped(img,box,margin=margin)
            aligned_images.append(aligned)
            boxs.append(box)
            imgs.append(img)
            
    return np.array(aligned_images),boxs,imgs


def calc_embs(filepaths, margin=10, batch_size=1):
    aligned_images , boxs , imgs = load_and_align_images(filepaths, margin)
    aligned_images = prewhiten(aligned_images)
    pd = []
    for start in range(0, len(aligned_images), batch_size):
        pd.append(model.predict_on_batch(aligned_images[start:start+batch_size]))
    embs = l2_normalize(np.concatenate(pd))

    return embs,boxs,imgs

def main():
    parser = argparse.ArgumentParser(description='Process extract embedding image.')

    parser.add_argument('image_path', type=str,help='folder of images')
    parser.add_argument('output', type=str,help='output_folder of embddings')
    # parser.add_argument('-m','--mode', type=str,help='output_folder of embddings')

    args = parser.parse_args()
    image_path = args.image_path
    output_path = args.output

    # mode = args.mode

    for folder in os.listdir(image_path):

        sub = image_path+"/"+folder
        label = os.path.basename(sub)

        filepaths = [os.path.join(sub,f) for f in os.listdir(sub)]

        print(label)
        embds,_,_ = calc_embs(filepaths,margin=10)

        print(embds.shape)

        with open(os.path.join(output_path,label+".pickle"),"wb") as pkl:
            pickle.dump(embds,pkl)


if __name__ == "__main__":
    main()