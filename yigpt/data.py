from torch.utils.data import Dataset


class FileDataset(Dataset):

    def __init__(self, fn):
        with open(fn, encoding="utf-8") as f:
            self.text = f.read()

    