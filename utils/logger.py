import csv
import os.path
import matplotlib

matplotlib.use('Agg')  # 设置matplotlib的后端，agg跟写入到文件有关，该句必须写在Import之前

from matplotlib import pyplot as plt
import numpy as np

plt.switch_backend('agg')


class CsvLogger:
    def __init__(self, filepath='logs/', filename='results.csv', data=None):
        self.log_path = filepath
        self.log_name = filename
        self.csv_path = os.path.join(self.log_path, self.log_name)
        self.fieldsnames = ['epoch', 'train_loss', 'val_loss']

        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldsnames)
            writer.writeheader()

        self.data = {}
        for field in self.fieldsnames:
            self.data[field] = []
        if data is not None:
            for d in data:
                d_num = {}
                for key in d:
                    d_num[key] = float(d[key]) if key != 'epoch' else int(d[key])
                self.write(d_num)

    def write(self, data):
        for k in self.data:
            self.data[k].append(data[k])
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldsnames)
            writer.writerow(data)

    def plot_progress_loss(self, title='loss pregress'):
        plt.figure(figsize=(9, 8), dpi=100)
        plt.plot(self.data['train_loss'], label='Training')
        plt.plot(self.data['val_loss'], label='Valing')
        plt.title(title)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.xlim(0, len(self.data['train_loss']) + 1)
        plt.savefig(os.path.join(self.log_path, 'loss.png'))

    def plot_progress(self):
        self.plot_progress_loss()
        plt.close('all')
