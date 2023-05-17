import shutil
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import datetime
import copy

class SelectedDataset(Dataset):
    def __init__(self, data_dir):
        data = pd.read_csv(data_dir, header=None).to_numpy().astype(np.int)
        # np.random.shuffle(data)
        self.field = data[:,:-1]
        self.label = data[:,-1]
        self.field_dims = np.max(self.field, axis=0) + 1

    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):
        field = self.field[item]
        label = self.label[item]
        return field, label

def save_selected_data(save_path, selected_fields, selected_target):
    slct_fields = copy.deepcopy(selected_fields)
    slct_target = copy.deepcopy(selected_target)
    new_array = np.hstack((slct_fields,slct_target[:,None]))
    pd.DataFrame(new_array).to_csv(save_path, mode='a', header=None, index = None)
    del new_array, slct_fields, slct_target

def clear_selected_data(save_path):
    pd.DataFrame(data=None).to_csv(save_path, mode='w', header=None, index = None)

def backup_best_data(save_path):
    shutil.copy(save_path, save_path.replace('train', 'best'))

def record_excel(record_path, data_array, head1, head2):
    try:
        writer = pd.ExcelWriter(record_path, mode="a", engine="openpyxl")
    except:
        writer = pd.ExcelWriter(record_path, engine="openpyxl")
    data = pd.DataFrame(data_array)
    sheet_name = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
    data.to_excel(writer, sheet_name, float_format='%.8f', header=[head1, head2], index=False)
    writer.save()
    writer.close()
