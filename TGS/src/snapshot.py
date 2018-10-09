# modified based on https://github.com/titu1994/Snapshot-Ensembles/blob/master/snapshot.py
import numpy as np
import os

import keras.callbacks as callbacks
from keras.callbacks import Callback

class SnapshotModelCheckpoint(Callback):
    """Callback that saves the snapshot weights of the model.

    Saves the model weights on certain epochs (which can be considered the
    snapshot of the model at that epoch).

    Should be used with the cosine annealing learning rate schedule to save
    the weight just before learning rate is sharply increased.

    # Arguments:
        nb_epochs: total number of epochs that the model will be trained for.
        nb_snapshots: number of times the weights of the model will be saved.
        fn_prefix: prefix for the filename of the weights.
    """

    def __init__(self, nb_epochs, nb_snapshots, fn_prefix='Model'):
        super(SnapshotModelCheckpoint, self).__init__()

        self.check = nb_epochs // nb_snapshots
        self.fn_prefix = fn_prefix

    def on_epoch_end(self, epoch, logs={}):
        if epoch != 0 and (epoch + 1) % self.check == 0:
            filepath = "%s.snapshot.%d" % (self.fn_prefix, (epoch + 1) // self.check)
            self.model.save_weights(filepath, overwrite=True)
            print("Saved snapshot one %s at epoch %s" % (filepath, epoch))

class SnapshotCallbackBuilder:
    """Callback builder for snapshot ensemble training of a model.

    Creates a list of callbacks, which are provided when training a model
    so as to save the model weights at certain epochs, and then sharply
    increase the learning rate.
    """

    def __init__(self, nb_epochs, nb_snapshots, init_lr=0.1, monitor= 'val_acc', model_weight_prefix= ''):
        """
        Initialize a snapshot callback builder.

        # Arguments:
            nb_epochs: total number of epochs that the model will be trained for.
            nb_snapshots: number of times the weights of the model will be saved.
            init_lr: initial learning rate
        """
        self.T = nb_epochs
        self.M = nb_snapshots
        self.alpha_zero = init_lr
        self.monitor = monitor
        self.model_weight_prefix = model_weight_prefix

    def get_callbacks(self):
        """
        Creates a list of callbacks that can be used during training to create a
        snapshot ensemble of the model.

        Args:
            model_prefix: prefix for the filename of the weights.

        Returns: list of 3 callbacks [ModelCheckpoint, LearningRateScheduler,
                 SnapshotModelCheckpoint] which can be provided to the 'fit' function
        """
        if not os.path.exists('weights/'):
            os.makedirs('weights/')

        callback_list = [callbacks.ModelCheckpoint('%s.best' % self.model_weight_prefix,
                                                   monitor= self.monitor,
                                                   save_best_only=True,
                                                   save_weights_only=True,
                                                   mode= 'max',
                                                   verbose= 1),
                         callbacks.LearningRateScheduler(schedule=self._cosine_anneal_schedule, verbose= 1),
                         SnapshotModelCheckpoint(self.T, self.M, fn_prefix= self.model_weight_prefix)]

        return callback_list

    def _cosine_anneal_schedule(self, t):
        cos_inner = np.pi * (t % (self.T // self.M))  # t - 1 is used when t has 1-based indexing.
        cos_inner /= self.T // self.M
        cos_out = np.cos(cos_inner) + 1
        return float(self.alpha_zero / 2 * cos_out)
