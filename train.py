import argparse
import collections
import numpy as np
import torch
from torch import nn, optim
from parse_config import ConfigParser
import random
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from trainer import Trainer

def main(config: ConfigParser):

    logger = config.get_logger('train')

    data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        batch_size= config['data_loader']['args']['batch_size'],
        shuffle=config['data_loader']['args']['shuffle'],
        validation_split=config['data_loader']['args']['validation_split'],
        num_batches=config['data_loader']['args']['num_batches'],
        training=True,
        num_workers=config['data_loader']['args']['num_workers'],
        pin_memory=config['data_loader']['args']['pin_memory'] 
    )

    # valid_data_loader = None
    valid_data_loader = data_loader.split_validation(bs=128)

    # test_data_loader = None

    test_data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        batch_size=128,
        shuffle=False,
        training=False,
        num_workers=2
    ).split_validation(bs=128)


    # build model architecture, then print to console
    model = config.initialize('arch', module_arch)

    
    train_loss=getattr(module_loss, config['train_loss'])
    val_loss = getattr(module_loss, config['val_loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']] # top 1 and 5

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())

    optimizer = config.initialize('optimizer', torch.optim, [{'params': trainable_params}])

    lr_scheduler = config.initialize('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    trainer = Trainer(model, train_loss, metrics, optimizer,
                      config=config,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      test_data_loader=test_data_loader,
                      lr_scheduler=lr_scheduler,
                      val_criterion=val_loss)

    trainer.train()
    logger = config.get_logger('trainer', config['trainer']['verbosity'])

    return trainer.best_test_acc

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default='config_mnist.json', type=str,
                      help='config file path (default: None)')
    args.add_argument('-d', '--device', default='0', type=str,
                      help='indices of GPUs to use (default: 0)')

    # custom options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--fnr', '--feature_noise_rate'], type=float, target=('feature_noise_rate',)),
        CustomArgs(['--lnr', '--label_noise_rate'], type=float, target=('trainer','percent',)),
        CustomArgs(['--lr', '--learning_rate'], type=float, target=('optimizer', 'args', 'lr')),
        CustomArgs(['--bs', '--batch_size'], type=int, target=('data_loader', 'args', 'batch_size')),
        CustomArgs(['--asym', '--asym'], type=bool, target=('trainer', 'asym')),
        CustomArgs(['--name', '--exp_name'], type=str, target=('name',)),
        CustomArgs(['--seed', '--seed'], type=int, target=('seed',)),
        CustomArgs(['--arch', '--arch'], type=str, target=('arch','type')),
        CustomArgs(['--mo', '--monitor'], type=str, target=('trainer','monitor')), # {"min loss","min val_loss"}, default:"min loss", change this to "min val_loss" to use early stopping.
        CustomArgs(['--es', '--es'], type=int, target=('trainer', 'early_stop')),  # monitor stopping epochs
    ]
    config = ConfigParser.get_instance(args, options)
    random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    acc_lst=[]
    for rep in range(1):
        acc_lst.append(main(config))
    file_object = open('./results/result.txt', mode='a')
    file_object.writelines([config['name'],'_ln',str(config['trainer']['percent']), '\t','%.3f %.3f' % (np.mean(acc_lst), np.std(acc_lst)),  '\n'])
    file_object.close



