from keras.models import load_model
import pickle
import cv2,time
import numpy as np
from mtcnn.mtcnn import MTCNN

def _load_():
    facenet_model,mtcnn_model,image_size,encoder,clf = None,None,None,None,None
    # try:
    facenet_model = load_model("model/facenet_keras.h5")
    mtcnn_model = MTCNN(scale_factor=0.8,min_face_size=70,steps_threshold=[0.8]*3)
    image_size = 160
    encoder = pickle.load(open("model/encoder.pickle","rb"))
    clf = pickle.load(open("model/svc_model.pickle","rb"))
    
    return facenet_model,mtcnn_model,image_size,encoder,clf
    # except:
    #     return [None]*5

def calc_embs(model,aligned_images,label=None,batch_size=1):

    if model is None:
        return
        
    aligned_images = prewhiten(aligned_images)

    pd = []
    for start in range(0, len(aligned_images), batch_size):
        pd.append(model.predict_on_batch(aligned_images[start:start+batch_size]))
    embs = l2_normalize(np.concatenate(pd))

    return embs

def l2_normalize(x, axis=-1, epsilon=1e-10):
    output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
    return output

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

def infer(le, clf,facenet,aligned_images):
    """
    le : labels Encoder
    clf : svc model
    facenet : facenet model
    aligned_images.
    """
    embs = calc_embs(facenet,aligned_images)

    pred = le.inverse_transform(clf.predict(embs))
    proba = clf.predict_proba(embs)
    return pred,proba


facenet_model,mtcnn_model,image_size,encoder,clf = _load_()

cap = cv2.VideoCapture(0)
wd = cv2.namedWindow("",cv2.WINDOW_FREERATIO)

i = 0
n_false = 0
n = 200

while cap.isOpened():
    ret,img = cap.read()
    copy = img.copy()
    if ret:
        faces = mtcnn_model.detect_faces(img)
        if len(faces) > 0:

            # i+=1
            box,mtcnn_score,_ = get_bounding_boxes(faces[0]) 
            print("mtcnn score : %.2f"%mtcnn_score)    

            x,y,w,h = box
            
            crop = cropped(img, box,margin=10)
            pred,proba = infer(encoder,clf,facenet_model,np.array([crop]))
            pred,score = pred[0],proba[0].max()

            if score > 0.4:
                # if pred != "ManVanNam":
                #     n_false+=1
                print(pred , score)
                cv2.putText(copy,pred + ",%.2f"%score,(x,y),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,255),2)
                cv2.rectangle(copy,(x,y),(x+w,y+h),(0,255,0),2)

        cv2.imshow(wd,copy)

        k = cv2.waitKey(20)
        if k == ord("q"):
            break

    else:
        break

# print("score : %.2f"%(1-n_false/n))

cap.release()
cv2.destroyAllWindows()

