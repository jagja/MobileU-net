"""
Input is image files created by dat_gen.py
Creates a U-net style architecures with a pretrained encoder
Decoder is custom built with adjustable parameters
"""

import os
import glob
from itertools import product
import numpy as np
import cv2
import imageio

from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Conv2D, Concatenate, UpSampling2D, \
                         SpatialDropout2D, BatchNormalization, \
                         Activation, LeakyReLU
from keras.optimizers import Adam
from keras.models import Model
from keras.regularizers import l2
from keras import applications


# Useful function for decoder
def conv_block(tensor, nfilters, size=3, padding='same', alpha=0.3, w_decay=1.0):
    """ Constructs a convolutional block for use with decoder"""
    p = Conv2D(filters=nfilters,
               kernel_size=(size, size),
               padding=padding,
               kernel_regularizer=l2(w_decay))(tensor)

    p = BatchNormalization()(p)
    p = LeakyReLU(alpha=alpha)(p)

    p = Conv2D(filters=nfilters,
               kernel_size=(size, size),
               padding=padding,
               kernel_regularizer=l2(w_decay))(p)

    p = BatchNormalization()(p)
    p = LeakyReLU(alpha=alpha)(p)

    return p


# Define some hyperparameters
rgba = 4
decoder_blocks = 5
filters = 3
drop = 0.0
k_size = 3
lr = 4e-4
set_size = 64
val_size = 5*set_size
train_size = 20*set_size

batch_size = 32
i_size = 224
seed = 321
n_epochs = 10
opt = Adam(lr)

input_img = Input(shape=(224, 224, 3))


# Load pretrained encoder
encoder = applications.mobilenet_v2.MobileNetV2(input_tensor=input_img,
                                                alpha=0.35,
                                                include_top=False,
                                                weights='imagenet')

# Remove unwanted layers
encoder.layers.pop()
encoder.layers.pop()

# Fix encoder weights
# for l in encoder.layers:
#     l.trainable = False

x = encoder.layers[-1].output

recall = [119, 57, 30, 12, 0]
# Construct decoder
for i in range(decoder_blocks):
    x = conv_block(x, nfilters=filters * 2**(decoder_blocks-i-1))
    x = UpSampling2D((2, 2))(x)
    # Unwise to feed input directly into output
    x_recall = encoder.layers[recall[i]].output
    if recall[i] > 1:
        x = Concatenate()([x, x_recall])
    x = SpatialDropout2D(drop)(x)

x = Conv2D(1, (1, 1), padding="same")(x)

output_img = Activation("sigmoid")(x)
mobile_unet = Model(inputs=input_img, outputs=output_img)

# Check the structure
mobile_unet.summary()
mobile_unet.compile(optimizer=opt,
                    loss='binary_crossentropy',
                    metrics=['binary_accuracy'])

# Define some useful directories
base_dir = os.getcwd()
data_dir = os.path.join(base_dir, 'data2')
test_dir = os.path.join(base_dir, 'data2\\test\\*.png')

train_image_dir = os.path.join(base_dir, 'data2\\train\\train_images\\')
train_mask_dir = os.path.join(base_dir, 'data2\\train\\train_masks\\')
valid_image_dir = os.path.join(base_dir, 'data2\\valid\\valid_images\\')
valid_mask_dir = os.path.join(base_dir, 'data2\\valid\\valid_masks\\')

# Augment existing dataset
data_gen_args = dict(rescale=1./255,
                     rotation_range=15,
                     horizontal_flip=True,
                     vertical_flip=True)

train_datagen = ImageDataGenerator(**data_gen_args)
val_datagen = ImageDataGenerator(**data_gen_args)

train_image = train_datagen.flow_from_directory(directory=train_image_dir,
                                                target_size=(i_size, i_size),
                                                classes=None,
                                                class_mode=None,
                                                color_mode="rgb",
                                                seed=seed,
                                                batch_size=batch_size,
                                                shuffle=False)

train_mask = train_datagen.flow_from_directory(directory=train_mask_dir,
                                               target_size=(i_size, i_size),
                                               classes=None,
                                               class_mode=None,
                                               color_mode="grayscale",
                                               seed=seed,
                                               batch_size=batch_size,
                                               shuffle=False)

valid_image = val_datagen.flow_from_directory(directory=valid_image_dir,
                                              target_size=(i_size, i_size),
                                              classes=None,
                                              class_mode=None,
                                              color_mode="rgb",
                                              seed=seed,
                                              batch_size=batch_size,
                                              shuffle=False)

valid_mask = val_datagen.flow_from_directory(directory=valid_mask_dir,
                                             target_size=(i_size, i_size),
                                             classes=None,
                                             class_mode=None,
                                             color_mode="grayscale",
                                             seed=seed,
                                             batch_size=batch_size,
                                             shuffle=False)

train_generator = zip(train_image, train_mask)
val_generator = zip(valid_image, valid_mask)


# Learn weights
results = mobile_unet.fit_generator(train_generator,
                                    epochs=n_epochs,
                                    steps_per_epoch=train_size,
                                    validation_steps=val_size,
                                    validation_data=val_generator)


# Save weights
mobile_unet.save_weights('mu_weights.h5')

kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
# Save predictions
dim = (i_size, i_size)
scale = 8
pred_full = np.zeros((i_size*scale, i_size*scale))
count = 0
for image_path in glob.glob(test_dir):
    x_in = np.asarray(imageio.imread(image_path))
    for i, j in product(range(scale), range(scale)):
        window = x_in[32*i:32*(i+1), 32*j:32*(j+1), :rgba-1]

        x_mini = cv2.resize(window, dim, interpolation=cv2.INTER_CUBIC)
        x_mini = cv2.filter2D(x_mini, -1, kernel)

        pred_mini = mobile_unet.predict(x_mini[np.newaxis, :, :, :])
        pred_full[i_size*i:i_size*(i+1),
                  i_size*j:i_size*(j+1)] = pred_mini[0, :, :, 0]

    y_out = 255*pred_full
    mask_out = 255*(y_out > 128)
    imageio.imwrite('prob'+str(count)+'.png', y_out.astype(np.uint8))
    imageio.imwrite('mask'+str(count)+'.png', mask_out.astype(np.uint8))
    count += 1
