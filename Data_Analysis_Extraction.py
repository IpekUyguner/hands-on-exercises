import numpy as np  # To handle matrix operations
import cv2  # computer vision library
import dlib  # Automatic face tracking library


face_detector = (
    dlib.get_frontal_face_detector()
)  # instantiating face detector class from dlib library

#from keras.applications.xception import preprocess_input  # to preprocess the input
import warnings, os
import keras
import tensorflow as tf
from keras import layers, Model
from keras import models
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import (
    Dropout,
    InputLayer,
    Flatten,
    Dense,
    BatchNormalization,
    MaxPooling2D,
    Conv2D,
    Input,
    Concatenate,
    LeakyReLU,
)
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical, normalize
from keras.applications.xception import Xception, preprocess_input
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint, TensorBoard
import numpy as np
import os
from tqdm import tqdm_notebook as tqdm
import warnings
import keras.backend as K

warnings.filterwarnings("ignore")
import tensorflow as tf



warnings.filterwarnings("ignore")  #
from tqdm import tqdm


def get_boundingbox(face, width, height, scale=1.3, minsize=None):
    # Reference: https://github.com/ondyari/FaceForensics
    """
    Expects a dlib face to generate a quadratic bounding box.
    :param face: dlib face class
    :param width: frame width
    :param height: frame height
    :param scale: bounding box size multiplier to get a bigger face region
    :param minsize: set minimum bounding box size
    :return: x, y, bounding_box_size in opencv form
    """
    x1 = face.left()  # Taking lines numbers around face
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()
    size_bb = int(max(x2 - x1, y2 - y1) * scale)  # scaling size of box to 1.3
    if minsize:
        if size_bb < minsize:
            size_bb = minsize
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

    # Check for out of bounds, x-y top left corner
    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)
    # Check for too big bb size for given x, y
    size_bb = min(width - x1, size_bb)
    size_bb = min(height - y1, size_bb)

    return x1, y1, size_bb


DATASET_PATHS = {
    "original": "original_sequences",
    "FaceSwap": "original_sequences/faceswapkowalski_fake_videos",
}


# CHANGE NUMBERS!! TO DO:
train_original = (os.listdir(DATASET_PATHS["original"] + "/c40/images/"))[:3]
train_FaceSwap = (os.listdir(DATASET_PATHS["FaceSwap"] + "/c40/images/"))[:1]
train = [train_original, train_FaceSwap]
types = ["original", "FaceSwap"]


cv_original = (os.listdir(DATASET_PATHS["original"] + "/c40/images/"))[3:][:2]
cv_FaceSwap = (os.listdir(DATASET_PATHS["FaceSwap"] + "/c40/images/"))[1:][:1]
cv = [cv_original, cv_FaceSwap]

#test_original = (os.listdir(DATASET_PATHS["original"] + "/c40/images/"))[-11:-1]
#test_FaceSwap = (os.listdir(DATASET_PATHS["FaceSwap"] + "/c40/images/"))[-4:-1]
#t#est = [test_original, test_FaceSwap]


def track_face(split_type, Split, output_mkdir=True):
    """
    Expects a splited data list and generates face tracked images.
    :split_type: list video names for train/test/cv
    :Split: train/test/cv in str format
    This function will generate face tracked images for train/test/cv data
    and will place the same in corresponding directory
    """

    for part in zip(split_type, types):
        for video in tqdm(part[0]):
            if output_mkdir == True:
                os.makedirs(
                    "Data/" + Split + "/" + part[1] + "/" + video, exist_ok=True
                )
            input_path = DATASET_PATHS[part[1]] + "/c40/images/" + video
            output_path = "Data/" + Split + "/" + part[1] + "/" + video
            images = os.listdir(input_path)
            images.sort(key=lambda x: os.path.getmtime(input_path + "/" + x))
            for img in images[10:111]:  # Taking 101  frames from each video
                image = cv2.imread(input_path + "/" + img)
                faces = face_detector(image, 1)
                height, width = image.shape[:2]
                try:  # If in case face is not detected at any frame
                    x, y, size = get_boundingbox(
                        face=faces[0], width=width, height=height
                    )
                except IndexError:
                    continue
                cropped_face = image[y : y + size, x : x + size]
                cv2.imwrite(output_path + "/" + img, cropped_face)


#track_face(split_type=train,Split='train')
#track_face(split_type=test,Split='test')
#track_face(split_type=cv,Split='cv')

# Created to shuffle the videos
train_ = []
for ind, i in enumerate(train):
    for j in i:
        train_.append(j + "_" + types[ind])
test_ = []
#for ind, i in enumerate(test):
 #   for j in i:
  #      test_.append(j + "_" + types[ind])

cv_ = []
for ind, i in enumerate(cv):
    for j in i:
        cv_.append(j + "_" + types[ind])


TRAIN_DATADIR = "Data/train"
TEST_DATADIR = "Data/test"
CV_DATADIR = "Data/cv"


def create_data(DATADIR, shuffled_list):
    """
    Expects Data Directory and suffled videos list
    and will returns list of image arrays and its class label (1 or 0)
    return X,y => image_arrays, class_labels
    """
    data = []
    for name in shuffled_list:
        label = name.split("_")[-1]
        class_num = 0
        folder_name = name.split("_")[0]
        # seq = os.listdir(DATADIR+'/'+name.split('_')[-1])
        if len(name.split("_")) == 3:
            class_num = 1
            folder_name = name.split("_")[0] + "_" + name.split("_")[1]
        files = os.listdir(DATADIR + "/" + label + "/" + folder_name)
        path = DATADIR + "/" + label + "/" + folder_name
        files.sort(
            key=lambda x: os.path.getmtime(
                DATADIR + "/" + label + "/" + folder_name + "/" + x
            )
        )
        for img in tqdm(files):
            try:
                img_array = cv2.imread(os.path.join(path, img))
                img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
                new_array = cv2.resize(img_array, (299, 299))
                data.append([new_array, class_num])
            except Exception as e:
                print(str(e))

    print("Data gathering completed......\n Separating features and class lables")
    X = []
    y = []
    for row in tqdm(data):
        X.append(row[0])
        y.append(row[1])
    X = np.array(X).reshape(-1, 299, 299, 3)
    print("Done")
    return X, y


np.random.shuffle(train_)
x_train, y_train = create_data(DATADIR=TRAIN_DATADIR, shuffled_list=train_)
np.random.shuffle(test_)

#x_test, y_test = create_data(DATADIR=TEST_DATADIR, shuffled_list=test_)

x_cv, y_cv = create_data(DATADIR=CV_DATADIR, shuffled_list=cv_)

#####################

print("Shape of Train data {}".format(x_train.shape))
print("Shape of Cv data {}".format(x_cv.shape))
#print("Shape of Test data {}".format(x_test.shape))

Y_train = to_categorical(y_train)
#Y_test = to_categorical(y_test)
Y_cv = to_categorical(y_cv)


Xception_initial = Xception(
    include_top=False, weights="imagenet", input_shape=(299, 299, 3), pooling="avg",
)
# print(Xception_pre_trained.summary())

for layer in Xception_initial.layers:
    layer.trainable = True

x = Xception_initial.output
predicted = Dense(2, activation="softmax")(x)
model_pretrain = Model(input=Xception_initial.input, output=predicted)
model_pretrain.compile(
    loss=keras.losses.categorical_crossentropy,
    optimizer=keras.optimizers.Adam(lr=0.0002),
    metrics=["accuracy"],
)
pretraining_Xception = model_pretrain.fit(
    x_train, Y_train, verbose=1, batch_size=1, epochs=3
)

model_pretrain.layers.pop()  # Removing topmost layer
tensorboard = TensorBoard(log_dir="./logs", histogram_freq=3)
for layer in model_pretrain.layers:
    layer.trainable = False
x = model_pretrain.output
# x= Dense(64,activation ='relu')(x)
# x = Dropout(0.2)(x)
# x= Dense(64,activation ='relu')(x)
# x = Dropout(0.2)(x)
# x= BatchNormalization()(x)
predicted = Dense(2, activation="softmax")(x)
model_finetune1 = Model(input=model_pretrain.input, output=predicted)
model_finetune1.compile(
    loss=keras.losses.categorical_crossentropy,
    optimizer=keras.optimizers.Adam(lr=0.0002),
    metrics=["accuracy"],
)

finetuning1_XceptionNet = model_finetune1.fit(
    x_train,
    Y_train,
    verbose=1,
    batch_size=1,
    epochs=1,
    validation_data=(x_cv, Y_cv),
    callbacks=[tensorboard],
)

model_finetune1.save("model_finetuned_xception.hdf5")
