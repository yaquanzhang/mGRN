# to pre-process UEA datasets
from scipy.io import arff
import numpy as np
import pandas as pd
import os
import pathlib
import requests
import zipfile

def download(url: str, dest_folder: str):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)  # create folder if it does not exist

    filename = url.split('/')[-1].replace(" ", "_")  # be careful with file names
    file_path = os.path.join(dest_folder, filename)

    r = requests.get(url, stream=True)
    if r.ok:
        print("saving to", os.path.abspath(file_path))
        with open(file_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024 * 8):
                if chunk:
                    f.write(chunk)
                    f.flush()
                    os.fsync(f.fileno())
    else:  # HTTP status code 4XX/5XX
        print("Download failed: status code {}\n{}".format(r.status_code, r.text))


data_root_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), 'uea_datasets')
name_list = ['CharacterTrajectories',
 'FaceDetection',
 'LSST',
 'PenDigits',
 'PhonemeSpectra',
 'SpokenArabicDigits']
# download zip files from UEA websites:
# http://www.timeseriesclassification.com/dataset.php
# and save files in the folder uea_datasets
# Alternatively, you may download the zip files manually from the website and save them into uea_datasets folder
if not os.path.isdir(data_root_dir):
    os.makedirs(data_root_dir)
for name in name_list:
    print("downloading ", name)
    this_url = 'http://www.timeseriesclassification.com/Downloads/' + name + '.zip'
    download(this_url, data_root_dir)
# Extract the zip files
for name in name_list:
    print("Unzipping ", name)
    zip_file = os.path.join(data_root_dir, name + '.zip')
    this_data_dir = os.path.join(data_root_dir, name)
    if not os.path.isdir(this_data_dir):
        os.makedirs(this_data_dir)
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(this_data_dir)
# preprocessing the files and save into npy for future use.
for name in name_list:
    print(name)
    root_dir = os.path.join(data_root_dir, name)
    data_type_list = ['TRAIN', 'TEST']
    category_list = []
    for data_type in data_type_list:
        train_data_dir = os.path.join(root_dir, name + '_' + data_type + '.arff')
        data = arff.loadarff(train_data_dir)
        df = pd.DataFrame(data[0])
        print(df.columns)
        data_column = df.columns[0]
        label_columns = df.columns[1]
        data_list = []
        for ind, this_row in df[data_column].iteritems():
            data_list.append(np.array(this_row.tolist(), dtype = 'float32'))
        X = np.array(data_list)
        print(X.shape)
        X = np.transpose(X, (0, 2, 1)) # convert to (Sample, time, features)
        Y = df[label_columns].astype('category').cat.codes
        category_list.append(df[label_columns].astype('category').cat.categories)
        print(max(Y))
        np.savez(os.path.join(root_dir,data_type.lower()), X, Y)
    # to make sure the labels in train and test sets match with each other.
    check_category = category_list[0] == category_list[1]
    assert check_category.all()