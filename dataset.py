import numpy as np
import os
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


class CUB(Dataset):

    def __init__(
        self, 
        data_root, 
        image_size = 224,
        normalization = True,
        train_test = None
    ):
        
        self.data_root = data_root
        self.normalization = normalization
        self.get_attribite = False
        self.image_size = image_size
        self.train_test = train_test
        
        if self.normalization:
            if self.train_test == "train":
                self.transform = transforms.Compose(
                    [transforms.RandomHorizontalFlip(),
                     transforms.RandomVerticalFlip(),
                     transforms.RandomRotation(30),
                     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                     transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                     transforms.RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=2),
                     transforms.RandomPerspective(distortion_scale=0.2, p=0.5, interpolation=3, fill=0),
                     transforms.RandomGrayscale(p=0.2),
                     transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 2.0))], p=0.2),
                     transforms.ToTensor(),
                     transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                          std=[0.229, 0.224, 0.225])  # Normalize
                    ])
            else:
                self.transform = transforms.Compose(
                    [transforms.Resize((self.image_size, self.image_size)),
                     transforms.ToTensor(),
                     transforms.Normalize(mean = [0.485, 0.456, 0.406], 
                                          std = [0.229, 0.224, 0.225])
                    ])

        
        else:
            self.transform = transforms.Compose(
                [transforms.Resize((self.image_size, self.image_size)),
                 transforms.ToTensor(),
                 transforms.Normalize(mean = [0., 0., 0.], 
                                      std = [1., 1., 1.])
                ])

        
        # attribute list: a list of 312 attributes
        with open(os.path.join(self.data_root, 'attributes.txt'), 'r')  as f:
            self.attribute_list = f.readlines()
        self.attribute_list = [att.split()[1] for att in self.attribute_list]
        
        # image list: a list of 11788 images
        with open(os.path.join(self.data_root, './CUB_200_2011/images.txt'), 'r')  as f:
            self.image_list = f.readlines()
        self.image_list = [att.split()[1] for att in self.image_list]
        self.image_list = np.array(self.image_list)
        
        # class list: a list of 200 classes
        with open(os.path.join(self.data_root, './CUB_200_2011/classes.txt'), 'r')  as f:
            self.class_list = f.readlines()
        self.class_list = [att.split()[1] for att in self.class_list]
        
        # image to label: a list of the classes of the 11788 images
        with open(os.path.join(self.data_root, './CUB_200_2011/image_class_labels.txt'), 'r')  as f:
            self.image2label = f.readlines()
        self.image2label = np.array([int(att.split()[1])-1 for att in self.image2label])
        
        
        if self.train_test is not None:
            # train test split: a binary list of the is_trainset of 11788 images
            with open(os.path.join(self.data_root, './CUB_200_2011/train_test_split.txt'), 'r')  as f:
                train_test_split = f.readlines()
            train_test_split = np.array([int(att.split()[1]) for att in train_test_split])
            if self.train_test == 'train':
                self.indices = np.where(train_test_split)[0]
            else:
                self.indices = np.where(1-train_test_split)[0]
            self.image_list = self.image_list[self.indices]
            self.image2label = self.image2label[self.indices]

    def __len__(self):
        if self.train_test is None:
            return len(self.image_list)
        else:
            return len(self.indices)
        
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = Image.open(os.path.join(self.data_root, 'CUB_200_2011/images', self.image_list[idx]))
        image = image.convert('RGB')
        label = self.image2label[idx]
        image = self.transform(image)

        return image, label
import numpy as np
import os
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


class CUB(Dataset):

    def __init__(
        self, 
        data_root, 
        image_size = 224,
        normalization = True,
        train_test = None
    ):
        
        self.data_root = data_root
        self.normalization = normalization
        self.get_attribite = False
        self.image_size = image_size
        self.train_test = train_test
        
        if self.normalization:
            if self.train_test == "train":
                self.transform = transforms.Compose(
                    [transforms.RandomHorizontalFlip(),
                     transforms.RandomVerticalFlip(),
                     transforms.RandomRotation(30),
                     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                     transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                     transforms.RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=2),
                     transforms.RandomPerspective(distortion_scale=0.2, p=0.5, interpolation=3, fill=0),
                     transforms.RandomGrayscale(p=0.2),
                     transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 2.0))], p=0.2),
                     transforms.ToTensor(),
                     transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                          std=[0.229, 0.224, 0.225])  # Normalize
                    ])
            else:
                self.transform = transforms.Compose(
                    [transforms.Resize((self.image_size, self.image_size)),
                     transforms.ToTensor(),
                     transforms.Normalize(mean = [0.485, 0.456, 0.406], 
                                          std = [0.229, 0.224, 0.225])
                    ])

        
        else:
            self.transform = transforms.Compose(
                [transforms.Resize((self.image_size, self.image_size)),
                 transforms.ToTensor(),
                 transforms.Normalize(mean = [0., 0., 0.], 
                                      std = [1., 1., 1.])
                ])

        
        # attribute list: a list of 312 attributes
        with open(os.path.join(self.data_root, 'attributes.txt'), 'r')  as f:
            self.attribute_list = f.readlines()
        self.attribute_list = [att.split()[1] for att in self.attribute_list]
        
        # image list: a list of 11788 images
        with open(os.path.join(self.data_root, './CUB_200_2011/images.txt'), 'r')  as f:
            self.image_list = f.readlines()
        self.image_list = [att.split()[1] for att in self.image_list]
        self.image_list = np.array(self.image_list)
        
        # class list: a list of 200 classes
        with open(os.path.join(self.data_root, './CUB_200_2011/classes.txt'), 'r')  as f:
            self.class_list = f.readlines()
        self.class_list = [att.split()[1] for att in self.class_list]
        
        # image to label: a list of the classes of the 11788 images
        with open(os.path.join(self.data_root, './CUB_200_2011/image_class_labels.txt'), 'r')  as f:
            self.image2label = f.readlines()
        self.image2label = np.array([int(att.split()[1])-1 for att in self.image2label])
        
        
        if self.train_test is not None:
            # train test split: a binary list of the is_trainset of 11788 images
            with open(os.path.join(self.data_root, './CUB_200_2011/train_test_split.txt'), 'r')  as f:
                train_test_split = f.readlines()
            train_test_split = np.array([int(att.split()[1]) for att in train_test_split])
            if self.train_test == 'train':
                self.indices = np.where(train_test_split)[0]
            else:
                self.indices = np.where(1-train_test_split)[0]
            self.image_list = self.image_list[self.indices]
            self.image2label = self.image2label[self.indices]

    def __len__(self):
        if self.train_test is None:
            return len(self.image_list)
        else:
            return len(self.indices)
        
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = Image.open(os.path.join(self.data_root, 'CUB_200_2011/images', self.image_list[idx]))
        image = image.convert('RGB')
        label = self.image2label[idx]
        image = self.transform(image)

        return image, label
