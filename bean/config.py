config_lstm = {
    'batch_size': 64,
    'lstm_units': 300,
    'num_classes': 3,
    'n_epochs': 20,
    'embedding_size': 300,
    'l2_reg_lambda': 0.0,
    'clip': 5.0
}

config_cnn = {
    'n_epochs': 20,
    'kernel_sizes': [3, 4, 5],
    'dropout_rate': 0.5,
    'learning_rate': 0.01,
    'std_dev': 0.05,
    'n_filters': 100,
    'num_classes': 3,
    'l2_reg': 0.5,
    'max_sen_len': 30,
    'f_type_num': 3,
    'batch_size': 64,
    'filter_sizes': [3, 4, 5],
    'embedding_size': 300,
    'num_filters': 128,
    'l2_reg_lambda': 0.0
}
