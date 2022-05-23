import joblib
import json
import numpy as np
import base64
import cv2
from wavelet import w2d
import sklearn

__class_name_to_number = {}
__class_number_to_name = {}

__model = None


def classify_image(image_base64_data, file_path=None):
    imgs = get_cropped_image_if_2_eyes(file_path, image_base64_data)
    result = []

    # Model Building
    # for each image
    for img in imgs:
        scalled_raw_image = cv2.resize(img, (32,32))
        img_harr = w2d(img, 'db1',5)#Wvaelet Transform -> Stackoverflow
        # Scaleing & aCombining images both 
        scalled_img_harr= cv2.resize(img_harr,(32, 32))
        combined_img = np.vstack((scalled_raw_image.reshape(32*32*3,1), scalled_img_harr. reshape(32*32,1)))

        len_image_array = 32*32*3 + 32*32

        final_image = combined_img.reshape(1,len_image_array).astype(float) # for later APi
        result.append({
            'class': class_number_to_name(__model.predict(final_image)[0]),
            'probability': np.around(__model.predict_proba(final_image)*100,2).tolist()[0], # Chance %
            'dictionary_number': __class_name_to_number
        })

    return result

def class_number_to_name(class_num):
    return __class_number_to_name[class_num]
       
       
        # Loading Model
def load_saved_model():
    print("loading saved...starts")
    global __class_name_to_number
    global __class_number_to_name

    with open("./person_class_dict.json", "r") as f:
        __class_name_to_number = json.load(f)
        __class_number_to_name = {v:k for k,v in __class_name_to_number.items()}

    global __model
    if __model is None:
        with open('./ saved_model.pkl', 'rb') as f:
            __model = joblib.load(f)
    print("loading saved ......done..... huehue")

# base 64 -> cv2
def get_cv2_image_from_base64_string(b64str):
    '''
    credit: https://stackoverflow.com/questions/33754935/read-a-base-64-encoded-image-from-memory-using-opencv-python-library

    '''
    encoded_data = b64str.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


# done
def get_cropped_image_if_2_eyes(image_path, image_base64_data):
    face =cv2.CascadeClassifier('./opencv/haarcascades/haarcascade_frontalface_default.xml')
    eye =cv2.CascadeClassifier('./opencv/haarcascades/haarcascade_eye.xml')

    if image_path:
        img = cv2.imread(image_path)
    else:
        img = get_cv2_image_from_base64_string(image_base64_data)

    black_and_white = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces= face.detectMultiScale(black_and_white,1.3,5)

    cropped_faces=[]
    for (a,b,c,d) in faces:
        face_gray = black_and_white[b:b+d,a:a+c]
        face_color= img[b:b+d,a:a+c]        
        eyes = eye.detectMultiScale(face_gray)
        if (len(eyes)>=2 ):
            cropped_faces.append(face_color)
    return cropped_faces

#done
def get_b64_test_image_test():
    with open("b64.txt") as f:
        return f.read()

if __name__ == '__main__':
    load_saved_model()

    print(classify_image(get_b64_test_image_test(), None))
    print(classify_image(None, "./Testing_images/tam_test1.jpeg"))
    print(classify_image(None, "./Testing_images/tam_test2.jpeg"))
    print(classify_image(None, "./Testing_images/m1.jpeg"))
    print(classify_image(None, "./Testing_images/m2.jpeg"))
    
    print(classify_image(None, "./Testing_images/rd1.jpeg"))
    print(classify_image(None, "./Testing_images/rd2.jpeg"))
    print(classify_image(None, "./Testing_images/rd.jpeg"))




