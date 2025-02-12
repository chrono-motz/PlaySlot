"""
Modules for defining, initializing and orquestrating the execution of callbacks.
"""

import os
import sys
import inspect

from lib.logger import print_



class Callback:
    """
    Base class from which all callbacks inherit
    """

    def __init__(self, trainer):
        """ Callback initializer """
        pass

    def on_epoch_start(self, trainer):
        """ Called at the beginning of every epoch """
        pass

    def on_train_epoch_end(self, trainer):
        """ Called at the end of every training epoch """
        pass

    def on_valid_epoch_end(self, trainer):
        """ Called at the end of every validation epoch """
        pass

    def on_epoch_end(self, trainer):
        """ Called at the end of every epoch """
        pass

    def on_batch_start(self, trainer):
        """ Called at the beginning of every batch """
        pass

    def on_batch_end(self, trainer):
        """ Called at the end of every batch """
        pass

    def on_log_frequency(self, trainer):
        """ Called after every batch if current iter is multiple of log frequency """
        pass

    def on_image_log_frequency(self, trainer):
        """ Called after every batch if current iter is multiple of log frequency """
        pass



class Callbacks:
    """
    Module for registering callbacks from the callbacks file, and orquestrating
    calls to such callbacks
    """

    def __init__(self, trainer=None):
        """ Module initalizer """
        self.callbacks = []
        if trainer is not None:
            print_("Registering callbacks:")
            self.register_callbacks(trainer=trainer)
        return

    def register_callbacks(self, trainer):
        """ Loading callbacks from the callbacks file """
        # importing base callbacks
        print_("  --> Registering BASE callbacks")
        import base.base_callbacks as base_callbacks
        for name, cls in inspect.getmembers(base_callbacks, inspect.isclass):
            self._register_callback(name, cls)
        
        custom_callbacks_path = os.path.join(trainer.exp_path, "callbacks.py")
        if os.path.exists(custom_callbacks_path):
            print_("  --> Registering CUSTOM callbacks...")
            sys.path.append(trainer.exp_path)
            import callbacks  # importing callbacks.py file from experiment parameters
            for name, cls in inspect.getmembers(callbacks, inspect.isclass):
                self._register_callback(name, cls)
        else:
            print_("  --> No CUSTOM callbacks were provided...")
        return
        
    def _register_callback(self, name, cls):
        """ Registering a callback module """
        if issubclass(cls, Callback) and cls is not Callback:
            print_(f"    --> callback {name}: {cls.__name__}")
            self.callbacks.append(cls)
            return

    def initialize_callbacks(self, trainer):
        """ Initializing all callbacks """
        initialized_callbacks = []
        for callback in self.callbacks:
            initialized_callbacks.append(
                    callback(trainer=trainer)
                )
        self.callbacks = initialized_callbacks
        return
    
    def register_new_initialized_callback(self, callback):
        """ Registering an initialized callback object """
        if not isinstance(callback, Callback):
            raise TypeError(f"Callback of {type(callback)} must inherit from Callback")
        name = callback.__class__.__name__
        print_("  --> Manually registering an initialized callback...")
        print_(f"    --> callback {name}: {name}")
        self.callbacks.append(callback)
        return

    def on_epoch_start(self, trainer):
        """ Calling callbacks activated on epoch start """
        for callback in self.callbacks:
            callback.on_epoch_start(trainer=trainer)

    def on_epoch_end(self, trainer):
        """ Calling callbacks activated on epoch end """
        return_data = {}
        for callback in self.callbacks:
            out = callback.on_epoch_end(trainer=trainer)
            if out is not None:
                return_data = {**return_data, **out}
        return return_data

    def on_train_epoch_end(self, trainer):
        """ Calling callbacks activated on epoch start """
        for callback in self.callbacks:
            callback.on_train_epoch_end(trainer=trainer)

    def on_valid_epoch_end(self, trainer):
        """ Calling callbacks activated on epoch start """
        for callback in self.callbacks:
            callback.on_valid_epoch_end(trainer=trainer)

    def on_batch_start(self, trainer):
        """ Calling callbacks activated on batch start """
        for callback in self.callbacks:
            callback.on_batch_start(trainer=trainer)

    def on_batch_end(self, trainer):
        """ Calling callbacks activated on batch end """
        for callback in self.callbacks:
            callback.on_batch_end(trainer=trainer)

    def on_log_frequency(self, trainer):
        """ Calling callbacks activated on batch end """
        if(trainer.iter_ % trainer.exp_params["training"]["log_frequency"] == 0):
            for callback in self.callbacks:
                callback.on_log_frequency(trainer=trainer)

    def on_image_log_frequency(self, trainer):
        """ Calling callbacks activated on batch end """
        if(trainer.iter_ % trainer.exp_params["training"]["image_log_frequency"] == 0):
            for callback in self.callbacks:
                callback.on_image_log_frequency(trainer=trainer)


#
