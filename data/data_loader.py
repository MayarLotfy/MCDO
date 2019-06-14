import tensorflow as tf



def get_mnsit_data(size=30, variance_1=0.03, variance_2=0.03):
    mnist = tf.keras.datasets.mnist

    (x_train, y_train),(x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    return x_train , y_train , x_test,y_test