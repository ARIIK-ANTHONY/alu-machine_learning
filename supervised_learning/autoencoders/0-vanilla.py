#!/usr/bin/env python3
""" Vanilla Autoencoder
"""

import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model


def autoencoder(input_dims, hidden_layers, latent_dims):
    """ Creates an autoencoder

    Args:
        input_dims (int): dimensions of the model input
        hidden_layers (list): number of nodes for each hidden layer in encoder
        latent_dims (int): dimensions of the latent space representation

    Returns:
        encoder (Model): encoder model
        decoder (Model): decoder model
        auto (Model): full autoencoder model
    """
    # Encoder
    encoder_input = Input(shape=(input_dims,))
    x = encoder_input
    for nodes in hidden_layers:
        x = Dense(nodes, activation='relu')(x)
    encoder_output = Dense(latent_dims, activation='relu')(x)
    encoder = Model(encoder_input, encoder_output)

    # Decoder
    decoder_input = Input(shape=(latent_dims,))
    x = decoder_input
    for nodes in reversed(hidden_layers):
        x = Dense(nodes, activation='relu')(x)
    decoder_output = Dense(input_dims, activation='sigmoid')(x)
    decoder = Model(decoder_input, decoder_output)

    # Autoencoder
    auto_input = Input(shape=(input_dims,))
    encoded = encoder(auto_input)
    decoded = decoder(encoded)
    auto = Model(auto_input, decoded)
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
