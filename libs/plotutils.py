import matplotlib.pyplot as plt
import tensorflow as tf


def plot_loss(history: tf.keras.callbacks.History):
    plt.plot(history.history['loss'], label='Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoche')
    plt.ylabel('RMSE')
    plt.legend()
    plt.grid(True)
