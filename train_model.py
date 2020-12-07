# train a given model with given datasets
import torch
import numpy as np
import math
import torch.nn as nn
import time
import check_accuracy
import torch.utils.data
import os


def train_simulation(model, train_set, validation_X, validation_Y, batch_size, n_Y, num_epochs,
                     batch_first, learning_rate, early_stopping_delta_percentage,
                     early_stopping_patience):
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    n_minibatch = math.ceil(train_set.shape[0] / batch_size)
    torch.cuda.empty_cache()
    validation_loss_list = []
    n_feature = validation_X.shape[-1]
    for epoch in range(num_epochs):
        epoch_start = time.time()
        model.train()
        training_loss_list = []
        # shuffle train_set
        shulle_index = torch.randperm(train_set.size()[0])
        train_set = train_set[shulle_index]
        for i in range(n_minibatch):
            # dataset already in cuda
            minibatch = train_set[i * batch_size:(i + 1) * batch_size]
            minibatch_X = minibatch[:, :, 0:n_feature].float()
            minibatch_Y = minibatch[:, -1, -n_Y:].float() # regression
            if minibatch_Y.shape[-1]==1:
                minibatch_Y = minibatch_Y.view(minibatch_Y.shape[0])
            if not batch_first:
                minibatch_X = minibatch_X.permute(1, 0, 2)
            outputs = model(minibatch_X)
            if outputs.shape[1] == 1:
                outputs = outputs.view(outputs.shape[0])
            loss = criterion(outputs, minibatch_Y)
            training_loss_list.append(loss.item())
            # training_loss_list.append(1)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print(n_batch)
            # if (i+1) % 2 == 0:
            cumulative_loss = sum(training_loss_list) / len(training_loss_list)
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, time elapsed: {:.2f}'
                  .format(epoch + 1, num_epochs, i + 1, n_minibatch, cumulative_loss, time.time() - epoch_start),
                  end='\r')
        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, time elapsed: {:.2f}'
              .format(epoch + 1, num_epochs, i + 1, n_minibatch, cumulative_loss, time.time() - epoch_start))
        # Validation
        check_accuracy_is_print = True  # verbose
        validation_loss = check_accuracy.check_accuracy_single_model_simulation(
            model, validation_X, validation_Y, check_accuracy_is_print)
        validation_loss_list.append(validation_loss)
        ##### early stopping
        if epoch >= early_stopping_patience:
            benchamark_loss = validation_loss_list[-early_stopping_patience - 1]
            early_stopping_delta = benchamark_loss*early_stopping_delta_percentage
            check_list = [x >= benchamark_loss - early_stopping_delta for x in
                          validation_loss_list[-early_stopping_patience:]]
            if np.array(check_list).all():
                print("early stopping")
                break
    return model


def train_tapnet(model, X_train, y_train, X_test, y_test, this_model_saving_dir, batch_first,
                batch_size, learning_rate, training_epochs, save_model_starting_epoch, weight_decay):

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if batch_first:
        n_minibatch = math.ceil(X_train.shape[0] / batch_size)
    else:
        n_minibatch = math.ceil(X_train.shape[1] / batch_size)
    torch.cuda.empty_cache()
    validation_acc_list = []
    validation_loss_list = []
    for epoch in range(training_epochs):
        epoch_start = time.time()
        model.train()
        # shuffle the training sets
        if batch_first:
            shulle_index = torch.randperm(X_train.size(0))
            X_train = X_train[shulle_index]
        else:
            shulle_index = torch.randperm(X_train.size(1))
            X_train = X_train[:, shulle_index]
        y_train = y_train[shulle_index]
        # epoch begins
        training_loss_list = []
        for i in range(n_minibatch):
            if batch_first:
                minibatch_X = X_train[i * batch_size:(i + 1) * batch_size]
            else:
                minibatch_X = X_train[:, i * batch_size:(i + 1) * batch_size]
            minibatch_Y = y_train[i * batch_size:(i + 1) * batch_size]
            outputs = model(minibatch_X)
            if outputs.shape[1] == 1:
                outputs = outputs.view(outputs.shape[0])
            loss = criterion(outputs, minibatch_Y)
            training_loss_list.append(loss.item())
            # training_loss_list.append(1)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print(n_batch)
            # if (i+1) % 2 == 0:
            cumulative_loss = sum(training_loss_list) / len(training_loss_list)
            print('\rEpoch [{}/{}], Step [{}/{}], Loss: {:.4f}, time elapsed: {:.2f}'
                  .format(epoch + 1, training_epochs, i + 1, n_minibatch, cumulative_loss, time.time() - epoch_start),
                  end='')
        print('\rEpoch [{}/{}], Step [{}/{}], Loss: {:.4f}, time elapsed: {:.2f}'
              .format(epoch + 1, training_epochs, i + 1, n_minibatch, cumulative_loss, time.time() - epoch_start))
        # validation
        validation_loss, validation_acc = check_accuracy.check_accuracy_single_model_tapnet(
            model, X_test, y_test)
        validation_loss_list.append(validation_loss)
        validation_acc_list.append(validation_acc)
        max_acc = max(validation_acc_list)
        if validation_acc>= max_acc and epoch>save_model_starting_epoch:
            torch.save(model.state_dict(), this_model_saving_dir)
    return max_acc

def train_mimic3(model, train_data, val_data, this_model_saving_dir, criterion, check_accuracy_obj,
                learning_rate, weight_decay, lr_decay_loss, lr_decay_factor, num_epochs, save_metric,
                save_model_starting_epoch):
    '''
    :param model: the model to train
    :param this_model_saving_dir: the dir to save the trained model
    :param weight_decay: weight_decay of Adam
    :param lr_decay_loss: Below this loss, we decay the learning rate by lr_decay_factor.
                          This helps accelerate training in the first few steps.
    :param lr_decay_factor: learning rate decay factor
    :param save_metric: to save model based on which metric
    :param save_model_starting_epoch: to avoid saving models repetitively, models are only saved after the first
                                      save_model_starting_epoch number of epochs.
    :return: validation results history
    '''
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # num_epochs = 100 # this num of epochs is only tentative. The stopping point is determined by earily stoppting.
    n_minibatch =  train_data.__len__()
    validation_loss = 2
    is_adjusted = False
    validation_loss_list = []
    save_metric_list = []
    best_result = None
    for epoch in range(num_epochs):
        epoch_start = time.time()
        model.train()
        if (lr_decay_loss is not None) and (not is_adjusted) and validation_loss < lr_decay_loss:
            lr = learning_rate / lr_decay_factor
            print("learing rate is reset to", lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            is_adjusted = True
        training_loss_list = []
        for i, this_batch in enumerate(train_data):
            minibatch_X = this_batch[0]
            minibatch_Y = this_batch[1]
            outputs = model(minibatch_X)
            if outputs.shape[1] == 1:
                outputs = outputs.view(outputs.shape[0])
            loss = criterion(outputs, minibatch_Y)
            training_loss_list.append(loss.item())
            # training_loss_list.append(1)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print(n_batch)
            # if (i+1) % 2 == 0:
            cumulative_loss = sum(training_loss_list) / len(training_loss_list)
            print('\rEpoch [{}/{}], Step [{}/{}], Loss: {:.4f}, time elapsed: {:.2f}'
                  .format(epoch + 1, num_epochs, i + 1, n_minibatch, cumulative_loss, time.time() - epoch_start),
                  end='')
        print('\rEpoch [{}/{}], Step [{}/{}], Loss: {:.4f}, time elapsed: {:.2f}'
              .format(epoch + 1, num_epochs, i + 1, n_minibatch, cumulative_loss, time.time() - epoch_start))
        train_data.on_epoch_end()
        result_dict = check_accuracy_obj.check_accuracy(model, val_data)
        validation_loss = result_dict['loss']
        validation_loss_list.append(validation_loss)
        save_metric_list.append(result_dict[save_metric])
        if save_metric == 'loss':
            best_metric = min(save_metric_list)
        else:
            best_metric = max(save_metric_list)
        if best_metric == result_dict[save_metric]:
            best_result = result_dict
            if epoch >= save_model_starting_epoch:
                model_saving_str = os.path.join(this_model_saving_dir, 'model_weights.pth')
                torch.save(model.state_dict(), model_saving_str)
    return best_result


