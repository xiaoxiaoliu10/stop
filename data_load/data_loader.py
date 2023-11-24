import numpy as np
from torch.utils.data import Dataset

class My_Multivariate_Data(Dataset):
    def __init__(self, x, y, merge_dim=0):
        self.data_x = x
        self.data_y = y
        self.classes = self.__classes__()
        self.merge_dim = merge_dim

    def __getitem__(self, index):
        x = self.data_x[index, :, :]
        y = self.data_y[index]
        return x.astype(np.float32), y

    def __len__(self):
        return len(self.data_x)

    def __classes__(self):
        return self.data_y.max() + 1
