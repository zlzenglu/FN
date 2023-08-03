import sys

import numpy as np
from PIL import Image
import torchvision



def get_mnist(root, cfg_trainer, train=True,
                transform_train=None, transform_val=None,
                download=True, noise_file = ''):
    base_dataset = torchvision.datasets.MNIST(root, train=train, download=download)
    if train:
        train_idxs, val_idxs = train_val_split(base_dataset.targets)
        train_dataset = MNIST_train(root, cfg_trainer, train_idxs, train=True, transform=transform_train)
        val_dataset = MNIST_val(root, cfg_trainer, val_idxs, train=train, transform=transform_val)
        if cfg_trainer['asym']:
            train_dataset.asymmetric_noise()
            val_dataset.asymmetric_noise()
        else:
            train_dataset.symmetric_noise()
            val_dataset.symmetric_noise()
        
        print(f"Train: {len(train_dataset)} Val: {len(val_dataset)}")  # Train: 54000 Val: 6000
    else:
        train_dataset = []
        val_dataset = MNIST_val(root, cfg_trainer, None, train=train, transform=transform_val)
        print(f"Test: {len(val_dataset)}")
    
    
    
    return train_dataset, val_dataset


def train_val_split(base_dataset: torchvision.datasets.MNIST):
    num_classes = 10
    base_dataset = np.array(base_dataset)
    train_n = int(len(base_dataset) * 0.9 / num_classes)
    train_idxs = []
    val_idxs = []

    for i in range(num_classes):
        idxs = np.where(base_dataset == i)[0]
        np.random.shuffle(idxs)
        train_idxs.extend(idxs[:train_n])
        val_idxs.extend(idxs[train_n:])
    np.random.shuffle(train_idxs)
    np.random.shuffle(val_idxs)

    return train_idxs, val_idxs


class MNIST_train(torchvision.datasets.MNIST):
    def __init__(self, root, cfg_trainer, indexs, train=True,
                 transform=None, target_transform=None,
                 download=True):
        super(MNIST_train, self).__init__(root, train=train,
                                            transform=transform, target_transform=target_transform,
                                            download=download)
        self.num_classes = 10
        self.cfg_trainer = cfg_trainer
        self.train_data_ = self.data[indexs] #self.train_data_[indexs]
        self.train_labels_ = np.array(self.targets)[indexs]#np.array(self.train_labels_)[indexs]
        self.indexs = indexs
        self.prediction = np.zeros((len(self.train_data_), self.num_classes, self.num_classes), dtype=np.float32)
        self.noise_indx = []
        
    def symmetric_noise(self):
        self.train_labels_gt = self.train_labels_.copy()
        #np.random.seed(seed=888)
        indices = np.random.permutation(len(self.train_data_))
        for i, idx in enumerate(indices):
            if i < self.cfg_trainer['percent'] * len(self.train_data_):
                self.noise_indx.append(idx)
                self.train_labels_[idx] = np.random.randint(self.num_classes, dtype=np.int32)

    def asymmetric_noise(self):
        self.train_labels_gt = self.train_labels_.copy()
        for i in range(self.num_classes):
            indices = np.where(self.train_labels_ == i)[0]
            np.random.shuffle(indices)
            for j, idx in enumerate(indices):
                if j < self.cfg_trainer['percent'] * len(indices):
                    self.noise_indx.append(idx)
                    if i == 4:
                        self.train_labels_[idx] = 9
                    elif i == 7:
                        self.train_labels_[idx] = 1
                    elif i == 5:
                        self.train_labels_[idx] = 6
                    elif i == 8:
                        self.train_labels_[idx] = 3
                    elif i == 2:
                        self.train_labels_[idx] = 7
                
            

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target, target_gt = self.train_data_[index], self.train_labels_[index], self.train_labels_gt[index]


        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy())


        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img,target, index, target_gt

    def __len__(self):
        return len(self.train_data_)



class MNIST_val(torchvision.datasets.MNIST):

    def __init__(self, root, cfg_trainer, indexs, train=True,
                 transform=None, target_transform=None,
                 download=True):
        super(MNIST_val, self).__init__(root, train=train,
                                          transform=transform, target_transform=target_transform,
                                          download=download)

        # self.train_data_ = self.data[indexs]
        # self.train_labels_ = np.array(self.targets)[indexs]
        self.num_classes = 10
        self.cfg_trainer = cfg_trainer
        if train:
            self.train_data_ = self.data[indexs]
            self.train_labels_ = np.array(self.targets)[indexs]
        else:
            self.train_data_ = self.data
            self.train_labels_ = np.array(self.targets)
        self.train_labels_gt = self.train_labels_.copy()
    def symmetric_noise(self):
        
        indices = np.random.permutation(len(self.train_data_))
        for i, idx in enumerate(indices):
            if i < self.cfg_trainer['percent'] * len(self.train_data_):
                self.train_labels_[idx] = np.random.randint(self.num_classes, dtype=np.int32)

    def asymmetric_noise(self):
        for i in range(self.num_classes):
            indices = np.where(self.train_labels_ == i)[0]
            np.random.shuffle(indices)
            for j, idx in enumerate(indices):
                if j < self.cfg_trainer['percent'] * len(indices):
                    if i == 4:
                        self.train_labels_[idx] = 9
                    elif i == 7:
                        self.train_labels_[idx] = 1
                    elif i == 5:
                        self.train_labels_[idx] = 6
                    elif i == 8:
                        self.train_labels_[idx] = 3
                    elif i == 2:
                        self.train_labels_[idx] = 7
    def __len__(self):
        return len(self.train_data_)


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target, target_gt = self.train_data_[index], self.train_labels_[index], self.train_labels_gt[index]


        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy())


        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index, target_gt
        