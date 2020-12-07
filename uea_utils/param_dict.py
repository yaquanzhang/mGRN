
model_param_dict = {}

model_param_dict['CharacterTrajectories'] = {"model_name": 'mGRN', "n_rnn_units": 64, "batch_first":False,
                                 "input_size_list": 'total_split', "size_of": 4, "dropouti":0.1, "dropoutw":0,
                                 "dropouto":0.1}

model_param_dict['FaceDetection'] = {"model_name": 'mGRN', "n_rnn_units": 64, "batch_first":False,
                                 "input_size_list": [72]*2, "size_of": 2, "dropouti":0.0, "dropoutw":0,
                                 "dropouto":0.0}

model_param_dict['LSST'] = {"model_name": 'mGRN', "n_rnn_units": 32, "batch_first":False,
                                 "input_size_list": 'total_split', "size_of": 2, "dropouti":0.0, "dropoutw":0,
                                 "dropouto":0.0}

model_param_dict['PenDigits'] = {"model_name": 'mGRN', "n_rnn_units": 64, "batch_first":False,
                                 "input_size_list": 'total_split', "size_of": 2, "dropouti":0.0, "dropoutw":0,
                                 "dropouto":0.0}

model_param_dict['PhonemeSpectra'] = {"model_name": 'mGRN', "n_rnn_units": 32, "batch_first":False,
                                 "input_size_list": 'total_split', "size_of": 2, "dropouti":0.1, "dropoutw":0,
                                 "dropouto":0.0}

model_param_dict['SpokenArabicDigits'] = {"model_name": 'mGRN', "n_rnn_units": 64, "batch_first":False,
                                 "input_size_list": 'total_split', "size_of": 4, "dropouti":0.1, "dropoutw":0,
                                 "dropouto":0.5}

training_param_dict = {}
training_param_dict['CharacterTrajectories'] = {'batch_size':32, 'learning_rate': 1e-3, 'weight_decay': 1e-4,
                                       'training_epochs': 500,
                                       'save_model_starting_epoch': 100}


training_param_dict['FaceDetection'] = {'batch_size':128, 'learning_rate': 1e-3, 'weight_decay': 0,
                                       'training_epochs': 10,
                                       'save_model_starting_epoch': 1}

training_param_dict['LSST'] = {'batch_size':128, 'learning_rate': 1e-3, 'weight_decay': 1e-4,
                                       'training_epochs': 500,
                                       'save_model_starting_epoch': 100}

training_param_dict['PenDigits'] = {'batch_size':128, 'learning_rate': 1e-3, 'weight_decay': 1e-4,
                                       'training_epochs': 500,
                                       'save_model_starting_epoch': 100}

training_param_dict['PhonemeSpectra'] = {'batch_size':128, 'learning_rate': 1e-3, 'weight_decay': 1e-4,
                                       'training_epochs': 300,
                                       'save_model_starting_epoch': 50}

training_param_dict['SpokenArabicDigits'] = {'batch_size':128, 'learning_rate': 1e-3, 'weight_decay': 0,
                                       'training_epochs': 500,
                                       'save_model_starting_epoch': 50}
