import torch
from torch.utils.data import Dataset

class FunctionSampleDataset(Dataset):
    def __init__(self, points, values):
        """
        Args:
            points (array-like): Array of sample points.
            values (array-like): Array of sample values corresponding to the points.
        """
        self.points = torch.from_numpy(points).float()
        self.values = torch.from_numpy(values).float()

    def __len__(self):
        return len(self.points)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        #sample = {'point': self.points[idx], 'value': self.values[idx]}
        #return sample

        return self.points[idx], self.values[idx]