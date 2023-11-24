import pandas as pd
import os
import natsort
import numpy as np
from scipy.io.arff import loadarff

def extract_data(data):
    res_data = []
    res_labels = []
    for t_data, t_label in data:
        t_data = np.array([d.tolist() for d in t_data])
        t_label = t_label.decode("utf-8")
        res_data.append(t_data)
        res_labels.append(t_label)
    return np.array(res_data).swapaxes(1, 2), np.array(res_labels)

def arff_to_numpy(dataset):
    train_data = loadarff(f'datasets/Multivariate_arff/{dataset}/{dataset}_TRAIN.arff')[0]
    test_data = loadarff(f'datasets/Multivariate_arff/{dataset}/{dataset}_TEST.arff')[0]

    train_X, train_y = extract_data(train_data)
    test_X, test_y = extract_data(test_data)
    train_X = train_X.transpose(0, 2, 1)
    test_X = test_X.transpose(0, 2, 1)
    if not os.path.exists(f'datasets/UEA/{dataset}'):
        os.makedirs(f'datasets/UEA/{dataset}')
    np.save(f'datasets/UEA/{dataset}/{dataset}_train_x.npy', train_X)
    np.save(f'datasets/UEA/{dataset}/{dataset}_train_label.npy', train_y)
    np.save(f'datasets/UEA/{dataset}/{dataset}_test_x.npy', test_X)
    np.save(f'datasets/UEA/{dataset}/{dataset}_test_label.npy', test_y)


def arff_to_csv(data_set_dir, data_save_dir):
    data_set_list = os.listdir(data_set_dir)
    if not os.path.exists(f'datasets/UEA_csv'):
        os.makedirs(f'datasets/UEA_csv')
    for dataset_name in data_set_list:
        if dataset_name in ['DuckDuckGeese', 'FaceDetection', 'InsectWingbeat', 'JapaneseVowels']:
            arff_to_numpy(dataset_name)
        elif dataset_name in ['Descriptions','DataDimensions.csv']:
            continue
        else:
            dataset_name_path = data_set_dir + "/" + dataset_name
            if os.path.isdir(dataset_name_path):
                dataset_name_path_list = natsort.natsorted(os.listdir(dataset_name_path), alg=natsort.ns.PATH)
                train_label_tag = False
                test_label_tag = False
                for data_file in dataset_name_path_list:
                    data_format = data_file.split('.')[1]
                    data_name = data_file.split('.')[0]
                    if data_format == 'arff' and 'Dimension' in data_name:
                        train_or_test = data_name.split('_')[1].lower()
                        file_name = dataset_name_path + "/" + data_file
                        with open(file_name, encoding="utf-8") as f:
                            header = []
                            for line in f:
                                if line.startswith("@attribute"):
                                    header.append(line.split()[1])
                                elif line.startswith("@data"):
                                    break
                            if os.path.getsize(file_name) > 0:
                                data_label = pd.read_csv(f, header=None)
                            else:
                                print("---empty file---" + data_file)
                                continue
                            label = data_label.iloc[:, -1]
                            data = data_label.iloc[:, :data_label.shape[1] - 1]

                            data_csv_dir = data_save_dir + "/" + dataset_name
                            if not os.path.exists(data_csv_dir):
                                os.mkdir(data_csv_dir)

                            file_name_data = data_save_dir + "/" + dataset_name + "/" + data_name
                            file_name_label = data_save_dir + "/" + dataset_name + "/" + train_or_test + "_label.csv"

                            if not train_label_tag and train_or_test == 'train':
                                label.to_csv(file_name_label, mode='w', index=False, header=None, encoding='utf-8')
                                train_label_tag = True
                            if not test_label_tag and train_or_test == 'test':
                                label.to_csv(file_name_label, mode='w', index=False, header=None, encoding='utf-8')
                                test_label_tag = True

                            data.to_csv(file_name_data + ".csv", mode='w', index=False, header=None, encoding='utf-8')
                            # print(data_file)
        print(dataset_name,'done!')


arff_to_csv('datasets/Multivariate_arff', 'datasets/UEA_csv')