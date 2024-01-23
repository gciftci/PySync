import keras
from typing import List, Tuple, Any


class KerasModel:
    """
    A class for managing the machine learning model.

    This class encapsulates the creation, compilation, and configuration
    of a Keras model for processing data.

    Attributes:
        config (dict): Configuration settings for the model.
        class_names (List[str]): List of class names for the model output.
        model (keras.Model): The Keras model instance.
    """

    def __init__(self, config: dict, class_names: List[str]) -> None:
        """
        Initialize the KerasModel with configuration settings and class names.

        Args:
            config (dict): Configuration settings for the model.
            class_names (List[str]): List of class names for the model output.
        """
        self.config = config
        self.class_names = class_names
        self.model = self.generate_model()

    def residual_block(x, filters, conv_num=3, activation="relu") -> Any | None:
        # Shortcut
        s = keras.layers.Conv1D(filters, 1, padding="same")(x)
        for i in range(conv_num - 1):
            x = keras.layers.Conv1D(filters, 3, padding="same")(x)
            x = keras.layers.Activation(activation)(x)
        x = keras.layers.Conv1D(filters, 3, padding="same")(x)
        x = keras.layers.Add()([x, s])
        x = keras.layers.Activation(activation)(x)
        return keras.layers.MaxPool1D(pool_size=2, strides=2)(x)

    def generate_model(self) -> keras.Model:
        """
        Generate and compile the Keras model.

        This method constructs the model architecture and compiles it
        with the appropriate optimizer, loss function, and metrics.

        Returns:
            keras.Model: The compiled Keras model.
        """
        #  model building
        input_shape = (self.config['model']['general']['sampling_rate'] // 2, 1)
        num_classes = len(self.class_names)
        inputs = keras.layers.Input(shape=input_shape, name="input")

        x = self.residual_block(inputs, 16, 2)
        x = self.residual_block(x, 32, 2)
        x = self.residual_block(x, 64, 3)
        x = self.residual_block(x, 128, 3)
        x = self.residual_block(x, 128, 3)

        x = keras.layers.AveragePooling1D(pool_size=3, strides=3)(x)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(256, activation="relu")(x)
        x = keras.layers.Dense(128, activation="relu")(x)

        outputs = keras.layers.Dense(num_classes, activation="softmax", name="output")(x)

        model = keras.models.Model(inputs=inputs, outputs=outputs)


        return model
