DEFAULT_TRAINER_OPTIONS = {
    # rank to use for truncated SVD
    'svd_rank': 0,    # number of eigenvalues to accelerate
    'n_accel_eigs': 5,
    # constant to multiply the eigenvalues by (1 is no change from SGD update)
    'accel_const': 2.0,    # function to find the weights to freeze
    'weight_finder_func': None,
    # additional arguments to the weight finder func
    'weight_finder_func_args': [],
    # frequency (in steps) to write model, weights, and metadata
    'save_freq': 5,    # number of steps to train for
    'train_epochs': 1000,    # random seed (handled by the trainer)
    'random_seed': 0,
    # directory to write logs to. Will be created if doesn't exist
    'log_dir': 'logs',    # pytorch device the model is on
    'device': 'cuda',    # frequency (in mini-batches) to write loss to stdout
    'log_interval': 10,
    # mode to use for acceleration (stable, unstable, neutral)
    'acceleration_mode': 'stable',
    # for v5, number of steps to wait before calculating projection matrix.
    'projection_delay': 5,
    # for v6, length of history to save when calculating DMD
    'window_size': 5,
    # for v4. if true, don't roll back to best weights after predicting
    'only_predict': False,
}

DEFAULT_OPTIM_ARGS = {
    # rank to use for truncated SVD
    'svd_rank': 0,    # number of eigenvalues to accelerate
    'n_accel_eigs': 5,
    # constant to multiply the eigenvalues by (1 is no change from SGD update)
    'accel_const': 2.0,
    # mode to use for acceleration (stable, unstable, neutral)
    'acceleration_mode': 'stable',
    # small value to use when classifying eigenvalues as stable, unstable, or neutral
    'epsilon': 5e-3,    # level to log at
    'log_level': 'INFO',
    # for efficient kga, number of steps to wait before accelerating
    'projection_delay': 5,
}
