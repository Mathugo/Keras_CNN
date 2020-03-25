from keras.layers import Dense, Conv2D, MaxPool2D , Flatten
from keras.layers.core import Activation, Dense, Dropout, Flatten
from keras.applications.inception_v3 import InceptionV3
from keras.layers.pooling import GlobalAveragePooling2D
from keras.models import Model

class INCEPTIONV3:
    @staticmethod
    def build(n_classes=2, dim=224,freeze_layers=30,full_freeze='N'):
        model = InceptionV3(weights='imagenet',include_top=False)
        x = model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)

        out = Dense(n_classes,activation='softmax')(x)

        model_final = Model(input = model.input,outputs=out)

        if full_freeze != 'N':
            for layer in model.layers[0:freeze_layers]:
                layer.trainable = False

        return model_final