import keras
from keras import backend as K
from keras.layers.core import Dense, Activation
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Model
from keras.applications import imagenet_utils
from keras.layers import Dense, Flatten, Dropout
from keras.applications import MobileNet
from keras.applications.mobilenet import preprocess_input
import numpy as np
from keras.optimizers import Adam
from keras.utils import multi_gpu_model
import tensorflow as tf
from PIL import Image
from time import time
from sklearn.metrics import classification_report, confusion_matrix

label_class = [    'botol',
                   'garpu',
                   'gayung',
                   'gelas',
                   'jam tangan',
                   'kaos sikil',
                   'kursi',
                   'mangkok',
                   'mejo',
                   'piring',
                   'pulpen',
                   'sendal',
                   'sendok',
                   'sepatu',
                   'tas',
                   'tipi']
DATASET_DIR_TRAIN = 'dataset/split/train_set'
DATASET_DIR_VAL = 'dataset/split/val_set'
DATASET_DIR_TEST = 'dataset/split/test_set'
img_size = 224
num_train = 50
num_test = 25
num_val = 25
batch_size = 32



def prepare_image(file):
    img_path = ''
    img = image.load_img(img_path + file, target_size=(img_size, img_size))
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)

def load_image(img_path, show=False):

    img = image.load_img(img_path, target_size=(img_size, img_size))
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.                                      # imshow expects values in the range [0, 1]

    if show:
        plt.imshow(img_tensor[0])
        plt.axis('off')
        plt.show()

    return img_tensor


base_model=keras.applications.mobilenet.MobileNet(input_shape=(img_size, img_size, 3),
                                                                #alpha = 1.0,
                                                                #depth_multiplier = 1,
                                                                dropout = 0.001,
                                                                pooling='avg',
                                                                include_top = False,
                                                                weights = "imagenet",classes = 16)
x=base_model.output
#x = MaxPooling2D(kernel_size=(2,2), strides=(2,2))
x = Dropout(0.001, name='dropout')(x)  #drop=0.001
preds=Dense(16,activation='softmax')(x) #final layer with softmax activation

model=Model(inputs=base_model.input,outputs=preds)
last_few_layers = 50 #number of the last few layers to freeze
for i,layer in enumerate(model.layers):
    print(i,layer.name)

for layer in model.layers[:86]:
    layer.trainable=False
for layer in model.layers[86:]:
    layer.trainable=True

print("Num GP   Us Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

#paralleled_model=multi_gpu_model(model, gpus=1)
train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input)
validation_datagen=ImageDataGenerator(preprocessing_function=preprocess_input)
test_datagen=ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator=train_datagen.flow_from_directory(DATASET_DIR_TRAIN,
                                                 target_size=(img_size,img_size),
                                                 color_mode='rgb',
                                                 batch_size=batch_size,
                                                 class_mode='categorical', shuffle=True)

validation_generator=validation_datagen.flow_from_directory(DATASET_DIR_VAL,
                                                            target_size=(img_size,img_size),
                                                            color_mode='rgb',
                                                            batch_size=batch_size,
                                                            class_mode='categorical', shuffle=True)

test_generator=test_datagen.flow_from_directory(DATASET_DIR_TEST,
                                                target_size=(img_size,img_size),
                                                color_mode='rgb',
                                                batch_size=batch_size,
                                                class_mode='categorical', shuffle=True)

model.summary()
model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
# Adam optimizer
# loss function will be categorical cross entropy
# evaluation metric will be accuracy
tensor = keras.callbacks.TensorBoard(log_dir='logs',
                                    histogram_freq=1,
                                    profile_batch=2,
                                    write_graph=True,
                                    write_grads=False,
                                    write_images=True,
                                    embeddings_freq=0,
                                    embeddings_layer_names=None,
                                    embeddings_metadata=None,
                                    embeddings_data=None,
                                    update_freq='epoch')
# checkpoint
class checkPoint(keras.callbacks.Callback):

    def __init__(self, model):
         self.model_to_save = model

    def on_epoch_end(self, epoch, logs=None):
        self.model_to_save.save('model/model_at_epoch_%d.h5' % epoch)

checkpoint = checkPoint(model)

callbacks_list = [checkpoint,tensor]

step_size_train=50  #train_generator.n//train_generator.batch_size/8
model.fit_generator(generator=train_generator,
                    steps_per_epoch=50,
                    epochs=20,
                    validation_data=validation_generator,
                    validation_steps=50,
                    callbacks=callbacks_list)

model.save('model/mbnet75.h5')

#matrix confusion
Y_pred = model.predict_generator(validation_generator, verbose = True)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
print(confusion_matrix(validation_generator.classes, y_pred))
print('Classification Report')
print(classification_report(validation_generator.classes, y_pred, target_names=label_class))

#model.load_weights('model/mbnet75.h5')
#img = Image.open("dataset/test_set/botol/botol_026.jpg")
#a = np.array(img).reshape(1,128,128,3)
#score = model.predict(a, batch_size=32, verbose=0)
#print(score)
