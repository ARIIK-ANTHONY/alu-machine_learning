#!/usr/bin/env python3
""" Variational Autoencoder"""

import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    function that creates a variational autoencoder
    Args:
        input_dims: integer containing the dimensions of the model input
        hidden_layers: list containing the number of nodes for each hidden
                       layer in the encoder, respectively
        latent_dims: integer containing the dimensions of the latent space
                     representation
    Returns: encoder, decoder, auto
    """
    # Encoder
    encoder_input = keras.Input(shape=(input_dims,))
    x = encoder_input
    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation='relu')(x)

    z_mean = keras.layers.Dense(latent_dims, activation=None)(x)
    z_log_var = keras.layers.Dense(latent_dims, activation=None)(x)

    def sampling(args):
        """Sampling similar points in latent space"""
        z_mean, z_log_var = args
        batch = keras.backend.shape(z_mean)[0]
        dim = keras.backend.int_shape(z_mean)[1]
        epsilon = keras.backend.random_normal(shape=(batch, dim))
        return z_mean + keras.backend.exp(z_log_var / 2) * epsilon

    z = keras.layers.Lambda(sampling, output_shape=(latent_dims,))(
        [z_mean, z_log_var])
    encoder = keras.Model(encoder_input, [z, z_mean, z_log_var])

    # Decoder
    decoder_input = keras.Input(shape=(latent_dims,))
    x = decoder_input
    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(nodes, activation='relu')(x)
    decoder_output = keras.layers.Dense(input_dims, activation='sigmoid')(x)
    decoder = keras.Model(decoder_input, decoder_output)

    # Autoencoder
    auto_input = keras.Input(shape=(input_dims,))
    z, z_mean, z_log_var = encoder(auto_input)
    reconstructed = decoder(z)
    auto = keras.Model(auto_input, reconstructed)

    def vae_loss(x, x_decoder_mean):
        """VAE loss function combining reconstruction loss and KL divergence"""
        reconstruction_loss = keras.backend.binary_crossentropy(
            x, x_decoder_mean)
        reconstruction_loss = keras.backend.sum(reconstruction_loss, axis=-1)
        kl_loss = -0.5 * keras.backend.sum(
            1 + z_log_var - keras.backend.square(z_mean) -
            keras.backend.exp(z_log_var), axis=-1)
        return keras.backend.mean(reconstruction_loss + kl_loss)

    auto.compile(loss=vae_loss, optimizer='adam')
    return encoder, decoder, auto
