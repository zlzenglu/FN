import sys
from torchvision import datasets, transforms
from base import BaseDataLoader
from data_loader.mnist import get_mnist
from data_loader.fmnist import get_fmnist
from data_loader.miniimagenet import get_mini
from parse_config import ConfigParser



class MNISTDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_batches=0, training=True,
                 num_workers=4, pin_memory=True):
        config = ConfigParser.get_instance()
        cfg_trainer = config['trainer']

        transform_train = transforms.Compose([
            transforms.Resize([32, 32]),
            # transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            # transforms.Normalize((0.1307,), (0.3081,))
        ])

        transform_val = transforms.Compose([
            transforms.Resize([32, 32]),
            transforms.ToTensor(),
            # transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir

        
        self.train_dataset, self.val_dataset = get_mnist(config['data_loader']['args']['data_dir'], cfg_trainer,
                                                           train=training,
                                                           transform_train=transform_train, transform_val=transform_val)

        super().__init__(self.train_dataset, batch_size, shuffle, validation_split, num_workers, pin_memory,
                         val_dataset=self.val_dataset)


class MINIDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_batches=0,  training=True, num_workers=4,  pin_memory=True):
        config = ConfigParser.get_instance()
        cfg_trainer = config['trainer']
        
        if cfg_trainer['gray']:
            transform_train = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
            ])
            transform_val = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
            ])
        else:
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(),
            ])
            transform_val = transforms.Compose([
                transforms.ToTensor(),
            ])
        self.data_dir = data_dir

        self.train_dataset, self.val_dataset = get_mini(config['data_loader']['args']['data_dir'], cfg_trainer, train=training,
                                                           transform_train=transform_train, transform_val=transform_val, validation_split=validation_split)

        super().__init__(self.train_dataset, batch_size, shuffle, validation_split, num_workers, pin_memory,
                         val_dataset = self.val_dataset)
    def run_loader(self, batch_size, shuffle, validation_split, num_workers, pin_memory):
        super().__init__(self.train_dataset, batch_size, shuffle, validation_split, num_workers, pin_memory,
                         val_dataset = self.val_dataset)

class FMNISTDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_batches=0, training=True,
                 num_workers=4, pin_memory=True):
        config = ConfigParser.get_instance()
        cfg_trainer = config['trainer']

        transform_train = transforms.Compose([
            transforms.Resize([32, 32]),
            # transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            # transforms.Normalize((0.1307,), (0.3081,))
        ])

        transform_val = transforms.Compose([
            transforms.Resize([32, 32]),
            transforms.ToTensor(),
            # transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir


        self.train_dataset, self.val_dataset = get_fmnist(config['data_loader']['args']['data_dir'], cfg_trainer,
                                                           train=training,
                                                           transform_train=transform_train, transform_val=transform_val)

        super().__init__(self.train_dataset, batch_size, shuffle, validation_split, num_workers, pin_memory,
                         val_dataset=self.val_dataset)
