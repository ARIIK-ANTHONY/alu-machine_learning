#!/usr/bin/env python3
"""
Defines class NST that performs tasks for neural style transfer
"""


import numpy as np
import tensorflow as tf


class NST:
    """
    Performs tasks for Neural Style Transfer.

    Public class attributes:
        style_layers: list of style layer names
        content_layer: name of the content layer

    Instance attributes:
        style_image: preprocessed style image
        content_image: preprocessed content image
        alpha: weight for content cost
        beta: weight for style cost
        model: Keras model used to calculate features
        gram_style_features: list of gram matrices
        content_feature: content feature of the content image
    """

    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                    'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """
        Initializes the NST class.

        Args:
            style_image: numpy.ndarray containing the style image
            content_image: numpy.ndarray containing the content image
            alpha: weight for the content cost
            beta: weight for the style cost
        """
        if not isinstance(style_image, np.ndarray):
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)")

        if len(style_image.shape) != 3 or style_image.shape[2] != 3:
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)")

        if not isinstance(content_image, np.ndarray):
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)")

        if len(content_image.shape) != 3 or content_image.shape[2] != 3:
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)")

        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")

        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        tf.enable_eager_execution()

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta

        self.load_model()
        self.generate_features()

    @staticmethod
    def scale_image(image):
        """
        Rescales an image so its largest side is 512 pixels.

        The image is also scaled so its pixel values are between 0 and 1.

        Args:
            image: numpy.ndarray of shape (h, w, 3)

        Returns:
            Tensor containing the scaled image.
        """
        if not isinstance(image, np.ndarray):
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)")

        if len(image.shape) != 3 or image.shape[2] != 3:
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)")

        height = image.shape[0]
        width = image.shape[1]

        if height > width:
            new_height = 512
            new_width = int(width * 512 / height)
        else:
            new_width = 512
            new_height = int(height * 512 / width)

        image = np.expand_dims(image, axis=0)

        image = tf.image.resize_bicubic(
            image,
            size=(new_height, new_width)
        )

        image = image / 255.0
        image = tf.clip_by_value(image, 0.0, 1.0)

        return image

    def load_model(self):
        """
        Creates the model used to calculate neural style costs.

        The model is based on VGG19 with AveragePooling2D replacing
        MaxPooling2D.
        """
        vgg19 = tf.keras.applications.VGG19(
            include_top=False,
            weights='imagenet'
        )

        vgg19.save("VGG19_base_model")

        custom_objects = {
            'MaxPooling2D': tf.keras.layers.AveragePooling2D
        }

        vgg = tf.keras.models.load_model(
            "VGG19_base_model",
            custom_objects=custom_objects
        )

        style_outputs = []
        content_output = None

        for layer in vgg.layers:
            if layer.name in self.style_layers:
                style_outputs.append(layer.output)

            if layer.name == self.content_layer:
                content_output = layer.output

            layer.trainable = False

        outputs = style_outputs + [content_output]

        self.model = tf.keras.models.Model(
            vgg.input,
            outputs
        )

    @staticmethod
    def gram_matrix(input_layer):
        """
        Calculates the Gram matrix of a feature layer.

        Args:
            input_layer: tensor of rank 4

        Returns:
            Gram matrix of shape (1, c, c)
        """
        if not isinstance(input_layer, (tf.Tensor, tf.Variable)):
            raise TypeError(
                "input_layer must be a tensor of rank 4"
            )

        if len(input_layer.shape) != 4:
            raise TypeError(
                "input_layer must be a tensor of rank 4"
            )

        _, height, width, channels = input_layer.shape

        size = int(height * width)

        features = tf.reshape(
            input_layer,
            (size, channels)
        )

        gram = tf.matmul(
            features,
            features,
            transpose_a=True
        )

        gram = tf.expand_dims(
            gram,
            axis=0
        )

        gram = gram / tf.cast(size, tf.float32)

        return gram

    def generate_features(self):
        """
        Extracts the features used to calculate neural style cost.
        """
        vgg19 = tf.keras.applications.VGG19(
            include_top=False,
            weights='imagenet'
        )

        style_image = vgg19.preprocess_input(
            self.style_image * 255
        )

        content_image = vgg19.preprocess_input(
            self.content_image * 255
        )

        style_outputs = self.model(style_image)[:-1]
        content_output = self.model(content_image)[-1]

        gram_style_features = []

        for feature in style_outputs:
            gram_style_features.append(
                self.gram_matrix(feature)
            )

        self.gram_style_features = gram_style_features
        self.content_feature = content_output

    def layer_style_cost(self, style_output, gram_target):
        """
        Calculates the style cost for a single layer.

        Args:
            style_output: tensor of rank 4
            gram_target: target Gram matrix

        Returns:
            Style cost for the layer.
        """
        if not isinstance(style_output, (tf.Tensor, tf.Variable)):
            raise TypeError(
                "style_output must be a tensor of rank 4"
            )

        if len(style_output.shape) != 4:
            raise TypeError(
                "style_output must be a tensor of rank 4"
            )

        channels = style_output.shape[3]

        if not isinstance(gram_target, (tf.Tensor, tf.Variable)):
            raise TypeError(
                "gram_target must be a tensor of shape [1, {}, {}]"
                .format(channels, channels)
            )

        if len(gram_target.shape) != 3:
            raise TypeError(
                "gram_target must be a tensor of shape [1, {}, {}]"
                .format(channels, channels)
            )

        if (gram_target.shape[0] != 1 or
                gram_target.shape[1] != channels or
                gram_target.shape[2] != channels):
            raise TypeError(
                "gram_target must be a tensor of shape [1, {}, {}]"
                .format(channels, channels)
            )

        gram_style = self.gram_matrix(style_output)

        difference = tf.square(
            gram_style - gram_target
        )

        cost = tf.reduce_sum(difference)

        cost = cost / tf.cast(
            channels ** 2,
            tf.float32
        )

        return cost

    def style_cost(self, style_outputs):
        """
        Calculates the style cost for the generated image.

        Args:
            style_outputs: list of style layer outputs

        Returns:
            Total style cost.
        """
        length = len(self.style_layers)

        if type(style_outputs) is not list or len(style_outputs) != length:
            raise TypeError(
                "style_outputs must be a list with a length of {}"
                .format(length)
            )

        weight = 1 / length

        cost = 0

        for i in range(length):
            cost += weight * self.layer_style_cost(
                style_outputs[i],
                self.gram_style_features[i]
            )

        return cost
