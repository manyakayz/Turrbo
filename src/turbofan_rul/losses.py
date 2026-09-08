"""Custom loss function used by the v4 model. Must be importable both at training
time and at model-load time (Keras needs it registered as a custom_object)."""
import tensorflow as tf
import tensorflow.keras.backend as K


def asymmetric_huber_loss(y_true, y_pred, delta: float = 10.0, late_penalty: float = 2.0):
    """
    Huber loss with an asymmetric penalty: predicting RUL too HIGH (a "late" /
    over-optimistic prediction) is penalized `late_penalty`x more than predicting
    too LOW. This directly optimizes toward a lower NASA scoring-function value,
    since late predictions are the operationally dangerous failure mode.
    """
    error = y_pred - y_true
    abs_error = K.abs(error)
    huber = tf.where(
        abs_error <= delta,
        0.5 * K.square(error),
        delta * (abs_error - 0.5 * delta),
    )
    penalty = tf.where(error > 0, late_penalty * huber, huber)
    return K.mean(penalty)


CUSTOM_OBJECTS = {"asymmetric_huber_loss": asymmetric_huber_loss}
