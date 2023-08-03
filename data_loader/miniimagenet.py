import sys
import numpy as np
import torch
from MLclf import MLclf

def get_mini(root, cfg_trainer, train=True,
                transform_train=None, transform_val=None,
                download=True, validation_split=0.1):
    # MLclf.miniimagenet_download(Download=download) ## need to run for the first time
    ## 60k images in total
    train_data, val_data, test_data = MLclf.miniimagenet_clf_dataset(
        ratio_train=0.7, ratio_val=validation_split, seed_value=None, shuffle=True, transform=transform_train, save_clf_data=True)
    seed=123
    if train:
        train_dataset=MINI_train(train_data,cfg_trainer)
        val_dataset=MINI_val(val_data,cfg_trainer)
        train_dataset.change_class(seed=seed)
        val_dataset.change_class(seed=seed)
        if cfg_trainer['asym']:
            train_dataset.asymmetric_noise()
            val_dataset.asymmetric_noise()
        else:
            train_dataset.symmetric_noise()
            val_dataset.symmetric_noise()
        
        print(f"Train: {len(train_dataset)} Val: {len(val_dataset)}")  # Train: 45000 Val: 5000
    else:
        train_dataset = []
        val_dataset = MINI_val(test_data,cfg_trainer)
        val_dataset.change_class(seed=seed)
        print(f"Test: {len(val_dataset)}")
    
    
    
    return train_dataset, val_dataset



class MINI_train():
    def __init__(self, dataset,cfg_trainer):
        self.num_classes = cfg_trainer['num_classes']
        self.cfg_trainer = cfg_trainer
        self.train_data = dataset.tensors[0]
        self.train_labels = dataset.tensors[1]
        self.prediction = np.zeros((len(self.train_data), self.num_classes, self.num_classes), dtype=np.float32)
        self.noise_indx = []
        
    def change_class(self,seed):
        if self.num_classes != 100:
            classes=np.random.RandomState(seed).randint(0, 100, size=self.num_classes) 
            # print('using classes:',classes)
            flags_train = np.full((len(self.train_labels)), False)
            for c in classes:
                idx_train = np.where(self.train_labels == c)[0]
                flags_train[idx_train] = True
            self.train_data=self.train_data[flags_train]
            self.train_labels=self.train_labels[flags_train]

            # rearrange the class number
            for i,c in enumerate(classes):
                self.train_labels[np.where(self.train_labels == c)[0]]=i
            


    def symmetric_noise(self):
        self.train_labels_gt = self.train_labels.clone()
        #np.random.seed(seed=888)
        indices = np.random.permutation(len(self.train_data))
        for i, idx in enumerate(indices):
            if i < self.cfg_trainer['percent'] * len(self.train_data):
                self.noise_indx.append(idx)
                self.train_labels[idx] = np.random.randint(self.num_classes, dtype=np.int32)

    def asymmetric_noise(self):
        self.train_labels_gt = self.train_labels.clone()
        for i in range(self.num_classes):
            indices = np.where(self.train_labels == i)[0]
            np.random.shuffle(indices)
            for j, idx in enumerate(indices):
                if j < self.cfg_trainer['percent'] * len(indices):
                    self.noise_indx.append(idx)
                    # truck -> automobile
                    if i == 9:
                        self.train_labels[idx] = 1
                    # bird -> airplane
                    elif i == 2:
                        self.train_labels[idx] = 0
                    # cat -> dog
                    elif i == 3:
                        self.train_labels[idx] = 5
                    # dog -> cat
                    elif i == 5:
                        self.train_labels[idx] = 3
                    # deer -> horse
                    elif i == 4:
                        self.train_labels[idx] = 7
                
            

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target, target_gt = self.train_data[index], self.train_labels[index], self.train_labels_gt[index]


        return img,target, index, target_gt

    def __len__(self):
        return len(self.train_data)



class MINI_val():

    def __init__(self, dataset,cfg_trainer):
        self.num_classes = cfg_trainer['num_classes']
        self.cfg_trainer = cfg_trainer
        self.train_data = dataset.tensors[0]
        self.train_labels = dataset.tensors[1].to(torch.int64)

        self.train_labels_gt = self.train_labels.clone()

    def change_class(self,seed):
        if self.num_classes != 100:
            classes=np.random.RandomState(seed).randint(0, 100, size=self.num_classes) 
            # print('using classes:',classes)
            flags_train = np.full((len(self.train_labels)), False)
            for c in classes:
                idx_train = np.where(self.train_labels == c)[0]
                flags_train[idx_train] = True
            self.train_data=self.train_data[flags_train]
            self.train_labels=self.train_labels[flags_train]

            # rearrange the class number
            for i,c in enumerate(classes):
                self.train_labels[np.where(self.train_labels == c)[0]]=i

    def symmetric_noise(self):
        
        indices = np.random.permutation(len(self.train_data))
        for i, idx in enumerate(indices):
            if i < self.cfg_trainer['percent'] * len(self.train_data):
                self.train_labels[idx] = np.random.randint(self.num_classes, dtype=np.int32)

    def asymmetric_noise(self):
        for i in range(self.num_classes):
            indices = np.where(self.train_labels == i)[0]
            np.random.shuffle(indices)
            for j, idx in enumerate(indices):
                if j < self.cfg_trainer['percent'] * len(indices):
                    # truck -> automobile
                    if i == 9:
                        self.train_labels[idx] = 1
                    # bird -> airplane
                    elif i == 2:
                        self.train_labels[idx] = 0
                    # cat -> dog
                    elif i == 3:
                        self.train_labels[idx] = 5
                    # dog -> cat
                    elif i == 5:
                        self.train_labels[idx] = 3
                    # deer -> horse
                    elif i == 4:
                        self.train_labels[idx] = 7
    def __len__(self):
        return len(self.train_data)


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target, target_gt = self.train_data[index], self.train_labels[index], self.train_labels_gt[index]



        return img, target, index, target_gt
        