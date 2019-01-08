import os
from skimage import io
from torch.utils.data import Dataset


class CelebADataset(Dataset):
    def __init__(self, df, source_dir, transform=None):
        self.images = df.to_dict()['image']
        assert os.path.isdir(source_dir), 'Image source dir is absent!'
        self.source_dir = source_dir
        self.transform = transform

    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, index):
        img_path = os.path.join(self.source_dir, self.images[index])
        image = io.imread(img_path)
        if self.transform is not None:
            image = self.transform(image)
        return image
        
        
        
