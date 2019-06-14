from matplotlib.backends.backend_pdf import PdfPages

import sys
sys.path.append("./")

from data.sample_generators import generate_linear_samples,generate_osband_sin_samples,generate_osband_nonlinear_samples
from data.data_loader import get_mnsit_data


import tensorflow as tf
import numpy as np
import plotting
import matplotlib.pyplot as plt

from training.combined_training import combined_training



def combined_evaluation(x_train, y_train,x_test,y_test, dropout, learning_rate, epochs, n_passes, ax):
    """

    :param x:`
    :param y:
    :param dropout:
    :param learning_rate:
    :param epochs:
    :param n_passes:
    :return:
    """
    sess, x_placeholder, dropout_placeholder = \
        combined_training(x_train, y_train, 0.2, learning_rate, epochs)

    prediction_op = sess.graph.get_collection("prediction")
    log_variance = sess.graph.get_collection("log_variance")
    aleatoric_op = tf.exp(log_variance)

    additional_range = 0.1 * np.max(x_train)
    x_eval = np.linspace(np.min(x_train) - additional_range, np.max(x_train) + additional_range, 100).reshape([-1, 1])

    feed_dict = {x_placeholder: x_train,
                 dropout_placeholder: dropout}

    predictions = []
    aleatorics = []
    for _ in range(n_passes):
        prediction, aleatoric = sess.run([prediction_op, aleatoric_op], feed_dict)
        predictions.append(prediction[0])
        aleatorics.append(aleatoric[0])

    y_eval = np.mean(predictions, axis=0).flatten()
    epistemic_eval = np.var(predictions, axis=0).flatten()
    aleatoric_eval = np.mean(aleatorics, axis=0).flatten()
    total_uncertainty_eval = epistemic_eval + aleatoric_eval

    plotting.plot_mean_vs_truth_with_uncertainties(x_train, y_train, x_test, y_test, aleatoric_eval, epistemic_eval, ax)

    fig.suptitle("Dropout - Learning Rate %f, Epochs %d, Dropout %.3f, Passes %d" %
                 (learning_rate, epochs, dropout, n_passes))

    # ax.fill_between(x_eval.flatten(), 0, epistemic_eval, label="epistemic", color="green", alpha=0.4)
    # ax.fill_between(x_eval.flatten(), 0, aleatoric_eval, label="aleatoric", color="orange", alpha=0.4)
    ax.legend()


if __name__ == "__main__":
    dropout_values = [0.1, 0.2, 0.3, 0.5]

    fig, axs = plt.subplots(len(dropout_values), 1, figsize=(30, 5*len(dropout_values)), sharey=True)
    axs[0].set_ylim([-1, 3])
    fig.suptitle('Combined-Model | Epochs: 15000, Learning Rate: 1e-3', fontsize=20)
    #x, y = generate_osband_sin_samples(60)
    x_train, y_train, x_test,y_test = get_mnsit_data(60)

    for dropout, ax in zip(dropout_values, axs):
        ax.set_title("%.3f Dropout" % dropout)
        combined_evaluation(x_train, y_train,x_test,y_test, dropout, 1e-3, 20000, 500, ax)
        fig.savefig("mnist.pdf")

    
