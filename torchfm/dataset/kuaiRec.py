import numpy as np
import pandas as pd
import torch


class KuaiRecDataset(torch.utils.data.Dataset):
    """
    KuaiRec Dataset

    Data preparation
        treat samples with a watch_ratio less than 1 as negative samples

    :param dataset_path: KuaiRec dataset path

    Reference:
        https://chongminggao.github.io/KuaiRec/
    """

    def __init__(self, dataset_path, sep=',', engine='c', header='infer'):
        data = pd.read_csv(dataset_path, sep=sep, engine=engine, header=header).to_numpy()
        self.items = data[:, :2].astype(np.int)
        self.targets = self.__preprocess_target(data[:, -1]).astype(np.float32)
        self.field_dims = np.max(self.items, axis=0) + 1
        self.user_field_idx = np.array((0, ), dtype=np.long)
        self.item_field_idx = np.array((1,), dtype=np.long)

    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, index):
        return self.items[index], self.targets[index]

    def __preprocess_target(self, target):
        target[target <= 1] = 0
        target[target > 1] = 1
        return target

# dataset_path = '../dataset/KuaiRec/data/small_matrix.csv'
# dataset = KuaiRecDataset(dataset_path)
# print(dataset.field_dims, len(dataset))
# print(pd.DataFrame(dataset.targets).value_counts())