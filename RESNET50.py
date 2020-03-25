from keras.layers import Dense, Conv2D, MaxPool2D , Flatten
from keras.layers.core import Activation, Flatten, Dense, Dropout
from keras.applications.resnet50 import ResNet50
from keras.layers.pooling import GlobalMaxPooling2D
from keras.models import Model

class RESNET50:
    @staticmethod
    def build(n_classes=2, width=224, height=224, depth=3):

        base_model = ResNet50(weights='imagenet', include_top=False)
        x = base_model.output
        x = GlobalMaxPooling2D()(x)
        x = Dropout(0.5)(x)
        x = Dense(100, activation="relu")(x)
        x = Dropout(0.5)(x)

        if n_classes == 1:
            x = Dense(n_classes, activation="sigmoid")(x)
        else:
            x = Dense(n_classes, activation="softmax")(x)

        base_model = Model(base_model.input, x, name="base_model")
        """
        if n_classes == 1:
            base_model.compile(loss="binary_crossentropy", metrics=['acc'], optimizer="adam")
        else:
            base_model.compile(loss="sparse_categorical_crossentropy", metrics=['acc'], optimizer="adam")
        """
        return base_model 