import sys
#sys.path.append("D:/GUC/Semester 8/TUM/MCDO/UncertaintyNN/")
sys.path.append("./")
print(sys.path)
from matplotlib.backends.backend_pdf import PdfPages
from data.sample_generators import generate_linear_samples,generate_osband_sin_samples,generate_osband_nonlinear_samples
from data.data_loader import get_mnsit_data
import numpy as np
import plotting
import matplotlib.pyplot as plt

from training.dropout_training import dropout_training


def dropout_evaluation(x, y, dropout, learning_rate, epochs, n_passes, ax):
    # Hardcoded training dropout
    sess, x_placeholder, dropout_placeholder = \
        dropout_training(x, y, 0.2, learning_rate, epochs)

    prediction_op = sess.graph.get_collection("prediction")

    additional_range = 0.1 * np.max(x)
    x_eval = np.linspace(np.min(x) - additional_range, np.max(x) + additional_range, 100).reshape([-1, 1])

    feed_dict = {x_placeholder: x_eval,
                 dropout_placeholder: dropout}

    predictions = []
    for _ in range(n_passes):
        predictions.append(sess.run(prediction_op, feed_dict)[0])

    y_eval = np.mean(predictions, axis=0).flatten()
    uncertainty_eval = np.var(predictions, axis=0).flatten()

    plotting.plot_mean_vs_truth(x, y, x_eval, y_eval, uncertainty_eval, ax)


if __name__ == "__main__":
    dropout_values = [0.1, 0.3, 0.5, 0.6]
    fig, axs = plt.subplots(len(dropout_values), 1, figsize=(30, 5*len(dropout_values)), sharey=True)
    #x, y = generate_osband_sin_samples()
    x_train, y_train, x_test,y_test = get_mnist_data()
    for dropout, ax in zip(dropout_values, axs):
        ax.set_title("%.3f Dropout" % dropout)
        dropout_evaluation(x, y, dropout, 1e-3, 20000, 100, ax)
        fig.savefig("Dropout_Sinus.pdf")

    fig, axs = plt.subplots(len(dropout_values), 1, figsize=(30, 5*len(dropout_values)), sharey=True)
    x, y = generate_osband_nonlinear_samples()
    for dropout, ax in zip(dropout_values, axs):
        ax.set_title("%.3f Dropout" % dropout)
        dropout_evaluation(x, y, dropout, 1e-3, 20000, 100, ax)
        fig.savefig("Dropout_Nonlinear.pdf")

