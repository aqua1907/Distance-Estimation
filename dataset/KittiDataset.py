from torch.utils.data.dataset import Dataset
import pandas as pd
import os


class KittiDataset(Dataset):
    def __init__(self, csv_file, root_dir):
        super(KittiDataset, self).__init__()

        self.df = pd.read_csv(csv_file)
        self.imgs = list(sorted(os.listdir(os.path.join(root_dir, "images"))))

    def __getitem__(self, item):
        pass

    def __len__(self):
        return len(self.imgs)
