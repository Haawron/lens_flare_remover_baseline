from PIL import Image
import torch.utils.data as data
from utils.utils import image_transform, image_transform_dacon
from tqdm import tqdm
import pickle
class Dataset(data.Dataset):
    def __init__(self,args=None):
        self.args = args
        self.train=self.args.train
        self.image_transform = image_transform_dacon  # image_transform
        self.train_path = f'{self.args.dir_data}/train_input_img_256'
        self.label_path = f'{self.args.dir_data}/train_label_img_256'
        self.data_len = 186940#100#self.args.data_size
    def __getitem__(self, idx):
        if self.args.train:
            with open(f'{self.train_path}/{idx}.pkl','rb') as f:
                _input = pickle.load(f)
            with open(f'{self.label_path}/{idx}.pkl','rb') as f:
                _target = pickle.load(f)
            _input,_target = self.image_transform(_input,_target)
            return _input,_target
        else:
            raise ('Not implementation')
    def __len__(self):
        return self.data_len



