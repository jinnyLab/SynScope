import tensorflow as tf
import tensorflow.keras as keras

class Val_Init(keras.callbacks.Callback):
    """Stop training when the loss is at its min, i.e. the loss stops decreasing.

  Arguments:
      patience: Number of epochs to wait after min has been hit. After this
      number of no improvement, training stops.
  """

    def __init__(self):
        super(Val_Init, self).__init__()
    def on_epoch_begin(self, epoch, logs=None):
        for tracker in self.model.get_tracker():
            tracker.reset_state()
    