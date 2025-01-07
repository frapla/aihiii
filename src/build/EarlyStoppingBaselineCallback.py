import warnings

import keras
from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.callbacks.callback import Callback
from keras.src.trainers import compile_utils
from keras.src.utils import io_utils


class EarlyStoppingBaseline(keras.callbacks.EarlyStopping):
    def __init__(
        self,
        monitor="val_loss",
        min_delta=0,
        patience=0,
        verbose=0,
        mode="auto",
        baseline=None,
        restore_best_weights=False,
        start_from_epoch=0,
    ):
        super().__init__(
            monitor=monitor,
            min_delta=min_delta,
            patience=patience,
            verbose=verbose,
            mode=mode,
            baseline=baseline,
            restore_best_weights=restore_best_weights,
            start_from_epoch=start_from_epoch,
        )

    def on_epoch_end(self, epoch, logs=None):
        if self.monitor_op is None:
            # Delay setup until the model's metrics are all built
            self._set_monitor_op()

        current = self.get_monitor_value(logs)
        if current is None or epoch < self.start_from_epoch:
            # If no monitor value exists or still in initial warm-up stage.
            return
        if self.restore_best_weights and self.best_weights is None:
            # If best weights were never set,
            # then the current weights are the best.
            self.best_weights = self.model.get_weights()
            self.best_epoch = epoch

        self.wait += 1
        if not self._is_improvement(current, self.baseline) and self.wait >= self.patience and epoch > 0:
            self.best = current
            self.best_epoch = epoch
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
            self.stopped_epoch = epoch
            self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            io_utils.print_msg(f"Epoch {self.stopped_epoch + 1}: early stopping as not reached baseline {self.baseline}")
        if self.restore_best_weights and self.best_weights is not None:
            if self.verbose > 0:
                io_utils.print_msg("Restoring model weights from " "the end of the best epoch: " f"{self.best_epoch + 1}.")
            self.model.set_weights(self.best_weights)
