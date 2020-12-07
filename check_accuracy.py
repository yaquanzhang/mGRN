# validation
import torch
import torch.nn as nn
import pandas as pd


def check_accuracy_single_model_tapnet(model, X_test, y_test, is_print = True):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        validation_outputs = model(X_test)
    if validation_outputs.shape[1] == 1:
        validation_outputs = validation_outputs.view(validation_outputs.shape[0])
    validation_loss = criterion(validation_outputs, y_test).item()
    Y_array_cpu = y_test.cpu().numpy()
    predictions_probability = nn.functional.softmax(validation_outputs, dim=1).cpu().numpy()
    predictions = predictions_probability.argmax(axis=1)
    validation_acc = (predictions == Y_array_cpu).sum() / Y_array_cpu.shape[0]
    if is_print:
        print("validation loss: {:.4f}".format(validation_loss))
        print("validation acc: {:.4f}".format(validation_acc))
    return validation_loss, validation_acc


def check_accuracy_single_model_simulation(model, X_array, Y_array, is_print):
    criterion = torch.nn.MSELoss()
    model.eval()
    with torch.no_grad():
        model_outputs = model(X_array)
    if model_outputs.shape[1] == 1:
        model_outputs = model_outputs.view(model_outputs.shape[0])
    loss = criterion(model_outputs, Y_array).item()
    if is_print:
        print("validation loss: {:.4f}".format(loss))
    del model_outputs
    torch.cuda.empty_cache()
    return loss


def check_accuracy_multiple_model_simulation(model_list, X_array, Y_array):
    is_print = False
    full_result = pd.DataFrame()
    for model in model_list:
        loss = check_accuracy_single_model_simulation(model, X_array, Y_array, is_print)
        full_result_dict = {"loss": loss}
        full_result = full_result.append(full_result_dict, ignore_index=True)
    return full_result


