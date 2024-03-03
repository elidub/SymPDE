from torch.utils.data import Dataset
import pytorch_lightning as pl
import torch

class BaseDataset(Dataset):
    def __init__(
            self,
            mode: str,
            task: str,
            data_params: dict = {},
            data_vars: dict = {},
            transform_params: dict = {},
            data_dir: str = '../data',
            N: int = -1,
    ):
        
        assert mode in ['train', 'test', 'val']
        raise NotImplementedError
    
    def __getitem__(self, index):
        raise NotImplementedError
    
    def __len__(self):
        raise NotImplementedError
    
class BaseDataModule(pl.LightningDataModule):
    def __init__(
            self,
            dataset: BaseDataset,
            task: str,
            data_params: dict = {},
            data_vars: dict = {},
            transform_params: dict = {},
            data_dir: str = '../data',
            batch_size: int = 1,
            num_workers: int = 1,
            n_splits: list = [-1, -1, -1],
            persistent_workers: bool = False,
    ):
        super().__init__()
        self.dataset = dataset
        self.task = task
        self.data_params = data_params
        self.data_vars = data_vars
        self.transform_params = transform_params
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers

        self.n_splits = {'train': n_splits[0], 'val': n_splits[1], 'test': n_splits[2]}

    def setup(self, stage=None):
        self.datasets = {
            split:
                self.dataset(
                    mode=split,
                    task=self.task,
                    data_params = self.data_params,
                    data_vars = self.data_vars,
                    transform_params = self.transform_params,
                    data_dir=self.data_dir,
                    N=n_split,
                )
            for split, n_split in self.n_splits.items()
        }
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.datasets['train'], batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,
            persistent_workers = self.persistent_workers,
        )
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.datasets['val'], batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
            persistent_workers = self.persistent_workers,
        )
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.datasets['test'], batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
            persistent_workers = self.persistent_workers,
        )
    
    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.datasets['test'], batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
            persistent_workers = self.persistent_workers,
        )