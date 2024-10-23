from tqdm import tqdm
import numpy as np
import torch


class DataHolder:
    def __init__(self, data):
        self.data = data
        #self.labels = np.load(labels_filename)
        #self.data = (self.data-np.mean(self.data))/(np.std(self.data))
        #self.labels = self.labels[90:]
        #indices = np.arange(len(self.data))  
        #np.random.shuffle(indices)       
        #self.data = self.data[indices]
        #self.labels = self.labels[indices]
        #np.random.shuffle(self.data)
        self.n = self.data.shape[0]
        self.good_data = self.data  # all the data starts out in the good-category
        self.bad_data = np.empty((0, len(self.data[0])), float)

    def remove_outliers(self, indices):
        if len(indices) == 0:
            return
        self.data_to_be_moved = self.good_data[np.array(indices)]
        self.good_data = np.delete(self.good_data, indices, axis=0)
        self.bad_data = np.append(self.bad_data, self.data_to_be_moved, axis=0)

    def add_inliers(self, indices):
        if len(indices) == 0:
            return
        self.data_to_be_moved = self.bad_data[np.array(indices)]
        self.bad_data = np.delete(self.bad_data, indices, axis=0)
        self.good_data = np.append(self.good_data, self.data_to_be_moved, axis=0)

    def get_good_data(self):
        return self.good_data

    def get_bad_data(self):
        return self.bad_data
    
    def get_data(self):
        return self.data

    #def get_labels(self):
    #    return self.labels

    def get_n(self):
        return self.n

    def get_n_good(self):
        return self.good_data.shape[0]

    def get_n_bad(self):
        return self.bad_data.shape[0]

    def get_n_features(self):
        return self.good_data.shape[-1]

    def save_good_data(self, dir):
        path = dir + "/good_data.npy"
        np.save(path, self.good_data)

    def save_bad_data(self, dir):
        path = dir + "/bad_data.npy"
        np.save(path, self.bad_data)


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return torch.tensor(self.data[index]).float(), index

class CustomDatasetLabel(torch.utils.data.Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return torch.tensor(self.data[index]).float(), self.label[index], index


def t2np(tensor):
    """pytorch tensor -> numpy array"""
    return tensor.cpu().data.numpy() if tensor is not None else None
