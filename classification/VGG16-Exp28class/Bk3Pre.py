import os
import h5py
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import ModelCheckpoint

#Define shape with number of colour chanels first
from keras import backend as bke
bke.set_image_dim_ordering('th')


pesosvgg16 = 'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'    #Weigths without neuon layer weights
clasificadorpath = 'BK3.h5'
img_width, img_height = 160, 120
train_data_dir = 'BK3'
nb_train_samples = 5538
test_data_dir = 'Test'
nb_test_samples = 512
nb_epoch = 100


def firstlayers():
    datagen = ImageDataGenerator(rescale=1./255, data_format='channels_first')

    model = Sequential()

    model.add(ZeroPadding2D((1, 1), input_shape=(3, img_width, img_height)))
    model.add(Conv2D(64, (3, 3), activation='relu', name='block1_conv1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu', name='block1_conv2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(128, (3, 3), activation='relu', name='block2_conv1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(128, (3, 3), activation='relu', name='block2_conv2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3), activation='relu', name='block3_conv1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3), activation='relu', name='block3_conv2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3), activation='relu', name='block3_conv3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu', name='block4_conv1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu', name='block4_conv2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu', name='block4_conv3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu', name='block5_conv1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu', name='block5_conv2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu', name='block5_conv3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))

    #Transfer Learning ImageNet-VGG16 (weights from convolutional layers)
    f = h5py.File(pesosvgg16)

    keysVGG16 = f.keys()    #VGG16 leyer names

    #Search equal name layers and assign weights to layer of my CNN
    for nlayerVGG in range(len(f.keys())) :         #traverse VGG16 weights layers
        nameLayer = keysVGG16[nlayerVGG]
        #print("VGG16 layer loop: " + nameLayer)
        for nlayerMine in range(len(model.layers)) :        #traverse my VGG16 structure
            #print("... searching layer: " + model.layers[nlayerMine].get_config()['name'])
            if nameLayer == model.layers[nlayerMine].get_config()['name'] :
                g = f[nameLayer]
                weights = [g[format(p)] for p in g.keys()]
                model.layers[nlayerMine].set_weights(weights)
                print("ASSIGN weights: from " + nameLayer + " ---> " + model.layers[nlayerMine].get_config()['name'])
                break

    f.close()   #Close VGG16 weights file reading

    print('Weights assigned')

    #TRAIN images cross conv-maxpool layer blocks and save
    generator = datagen.flow_from_directory(
            train_data_dir,
            target_size=(img_width, img_height),
            batch_size=32,
            class_mode=None,
            shuffle=False)
    firstlayerstrain = model.predict_generator(generator, nb_train_samples/32)
    print("convolution Train shape: " + repr(firstlayerstrain.shape))
    np.save(open('convolutionedTrainImages.npy', 'wb'), firstlayerstrain)

    #TEST images cross conv-maxpool layer blocks and save
    generatorTest = datagen.flow_from_directory(
            test_data_dir,
            target_size=(img_width, img_height),
            batch_size=32,
            class_mode=None,
            shuffle=False)
    firstlayersTest = model.predict_generator(generatorTest, nb_test_samples/32)
    print("convolution Test shape: " + repr(firstlayersTest.shape))
    np.save(open('convolutionedTestImages.npy', 'wb'), firstlayersTest)

    print('......Model and Data generation FINISHED')



def clasificador():
    print("Start function <<clasificador()>>")

    datagen = ImageDataGenerator(rescale=1./255, data_format='channels_first')
    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=32,
        class_mode='categorical',
        shuffle=False)

    #class indexes
    labels = []
    i = 0
    for _, y in generator:
        i += len(y)
        labels.append(y)
        if i == nb_train_samples:
            break
    labels = np.concatenate(labels)


    #read train images after convolutional layers
    train_data = np.load(open('convolutionedTrainImages.npy','rb'))
    print("Train data: " + repr(train_data.shape))
    print("Labels before: " + repr(labels.shape))
    train_labels = labels[:(nb_train_samples/32*32)]    #take integer number of images depends on batch size
    print("Labels after: " + repr(train_labels.shape))

    print(train_data.shape[1:])
    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(28, activation='softmax'))
    #model.add(Dense(102, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    #create callback to save best step model
    filepath="weightsBK3best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    print("Trining........")
    model.fit(train_data, train_labels, epochs=nb_epoch, callbacks=callbacks_list)

    #save model and weights
    model.save_weights(clasificadorpath)
    model_json = model.to_json()
    with open("modelBK3.json", "w") as json_file:
      json_file.write(model_json)

    print("Function finished <<clasificador()>>")



##### MAIN execution ######
firstlayers()
clasificador()
