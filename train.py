# System libs
import os
import time
# import math
import random
import argparse
from distutils.version import LooseVersion
# Numerical libs
import torch
import torch.nn as nn
import torch.utils.data as data
from augmentations import Compose, RandomSized, AdjustContrast, AdjustBrightness, RandomErasing, RandomVerticallyFlip, RandomHorizontallyFlip, RandomRotate, PaddingCenterCrop
# Our libs
from dataset import TrainDataset
from models import ModelBuilder, SegmentationModule
from utils import AverageMeter, parse_devices
from lib.nn import UserScatteredDataParallel, user_scattered_collate, patch_replication_callback
import lib.utils.data as torchdata

import numpy as np

# train one epoch
def train(segmentation_module, iterator, optimizers, history, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    ave_total_loss = AverageMeter()
    ave_acc = AverageMeter()
    ave_jaccard = AverageMeter()

    segmentation_module.train(not args.fix_bn)
    
    # main loop
    tic = time.time()
    for i in range(args.epoch_iters):
        batch_data = next(iterator)

        if torch.cuda.is_available():
            for item in batch_data:
                for tnsr in batch_data[item]:
                    tnsr = tnsr.cuda()
                if len(batch_data[item].shape) == 5:
                    batch_data[item] = batch_data[item].squeeze(0).cuda()
                #print(batch_data[k].shape)
        #for i in batch_data: #Randomly change brightness and contrast at runtime.
        #    i = random_contrast_brightness(i)

        data_time.update(time.time() - tic)

        batch_data["seg_label"] = batch_data["seg_label"].permute(1,0,2,3).squeeze(1).cuda()
    
        segmentation_module.zero_grad()
        # forward pass
        loss, acc = segmentation_module(batch_data)
        loss = loss.mean()
        
        jaccard = acc[1].float().mean()
        acc = acc[0].float().mean()
        
        # Backward
        loss.backward()
        for optimizer in optimizers:
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average loss and acc
        ave_total_loss.update(loss.data.item())
        ave_acc.update(acc.data.item()*100)
        ave_jaccard.update(jaccard.data.item()*100)

        # calculate accuracy, and display
        if i % args.disp_iter == 0:
            if args.unet==False:
                print('Epoch: [{}][{}/{}], Time: {:.2f}, Data: {:.2f}, '
                    'lr_encoder: {:.6f}, lr_decoder: {:.6f}, '
                    'Accuracy: {:4.2f}, Loss: {:.6f}'
                    .format(epoch, i, args.epoch_iters,
                            batch_time.average(), data_time.average(),
                            args.running_lr_encoder, args.running_lr_decoder,
                            ave_acc.average(), ave_total_loss.average()))
            else:
                print('Epoch: [{}][{}/{}], Time: {:.2f}, Data: {:.2f},'
                    ' lr_unet: {:.6f}, Accuracy: {:4.2f}, Jaccard: {:4.2f},  Loss: {:.6f}'
                    .format(epoch, i , args.epoch_iters,
                            batch_time.average(), data_time.average(),
                            args.running_lr_encoder, ave_acc.average(),
                            ave_jaccard.average(), ave_total_loss.average()))

            fractional_epoch = epoch - 1 + 1. * i / args.epoch_iters
            history['train']['epoch'].append(fractional_epoch)
            history['train']['loss'].append(loss.data.item())
            history['train']['acc'].append(acc.data.item())

        # adjust learning rate
        cur_iter = i + (epoch - 1) * args.epoch_iters
        adjust_learning_rate(optimizers, cur_iter, args)


def checkpoint(nets, history, args, epoch_num):
    print('Saving checkpoints...')
    if args.unet:
        (unet, crit) = nets
    else:
        (net_encoder, net_decoder, crit) = nets
    
    suffix_latest = 'epoch_{}.pth'.format(epoch_num)
    
    torch.save(history,
               '{}/history_{}'.format(args.ckpt, suffix_latest))
    if args.unet:
        dict_unet = unet.state_dict()
        torch.save(dict_unet,
                    '{}/unet_{}'.format(args.ckpt, suffix_latest))
    else:    
        dict_encoder = net_encoder.state_dict()
        dict_decoder = net_decoder.state_dict()

        torch.save(dict_encoder,
                   '{}/encoder_{}'.format(args.ckpt, suffix_latest))
        torch.save(dict_decoder,
                   '{}/decoder_{}'.format(args.ckpt, suffix_latest))

    # dict_encoder_save = {k: v for k, v in dict_encoder.items() if not (k.endswith('_tmp_running_mean') or k.endswith('tmp_running_var'))}
    # dict_decoder_save = {k: v for k, v in dict_decoder.items() if not (k.endswith('_tmp_running_mean') or k.endswith('tmp_running_var'))}


def group_weight(module):
    group_decay = []
    group_no_decay = []
    for m in module.modules():
        if isinstance(m, nn.Linear):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.conv._ConvNd):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.batchnorm._BatchNorm):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)

    assert len(list(module.parameters())) == len(group_decay) + len(group_no_decay)
    groups = [dict(params=group_decay), dict(params=group_no_decay, weight_decay=.0)]
    return groups


def create_optimizers(nets, args):
    if args.unet == False:
        (net_encoder, net_decoder, crit) = nets
        optimizer_encoder = torch.optim.SGD(
            group_weight(net_encoder),
            lr=args.lr_encoder,
            momentum=args.beta1,
            weight_decay=args.weight_decay)
        optimizer_decoder = torch.optim.SGD(
            group_weight(net_decoder),
            lr=args.lr_decoder,
            momentum=args.beta1,
            weight_decay=args.weight_decay)
        return (optimizer_encoder, optimizer_decoder)
    else:
        (unet, crit) = nets
        optimizer_unet = torch.optim.SGD(
            group_weight(unet),
            lr = args.lr_encoder,
            momentum=args.beta1,
            weight_decay=args.weight_decay)
        return [optimizer_unet]

def adjust_learning_rate(optimizers, cur_iter, args):
    scale_running_lr = ((1. - float(cur_iter) / args.max_iters) ** args.lr_pow)
    args.running_lr_encoder = args.lr_encoder * scale_running_lr
    args.running_lr_decoder = args.lr_decoder * scale_running_lr
    
    if args.unet == False:
        (optimizer_encoder, optimizer_decoder) = optimizers
        for param_group in optimizer_encoder.param_groups:
            param_group['lr'] = args.running_lr_encoder
        for param_group in optimizer_decoder.param_groups:
            param_group['lr'] = args.running_lr_decoder
    else:
        optimizer_unet = optimizers[0]
        for param_group in optimizer_unet.param_groups:
            param_group['lr'] = args.running_lr_encoder

def main(args):
    # Network Builders
    builder = ModelBuilder()
    net_encoder=None
    net_decoder=None
    unet=None
    
    if args.unet == False:
        net_encoder = builder.build_encoder(
            arch=args.arch_encoder,
            fc_dim=args.fc_dim,
            weights=args.weights_encoder)
        net_decoder = builder.build_decoder(
            arch=args.arch_decoder,
            fc_dim=args.fc_dim,
            num_class=args.num_class,
            weights=args.weights_decoder)
    else:
        unet = builder.build_unet(num_class=args.num_class, 
            arch=args.unet_arch,
            weights=args.weights_unet)

        print("Froze the following layers: ")
        for name, p in unet.named_parameters():
            if p.requires_grad == False:
                print(name)

    crit = nn.NLLLoss()
    #crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(50))
    #crit = nn.CrossEntropyLoss().cuda()
    #crit = nn.BCELoss()

    if args.arch_decoder.endswith('deepsup') and args.unet == False:
        segmentation_module = SegmentationModule(
            net_encoder, net_decoder, crit, args.deep_sup_scale)
    else:
        segmentation_module = SegmentationModule(
            net_encoder, net_decoder,  crit, is_unet=args.unet, unet=unet)

    train_augs = Compose([RandomSized(224), RandomHorizontallyFlip(), RandomVerticallyFlip(), RandomRotate(180), AdjustContrast(cf=0.25), AdjustBrightness(bf=0.25)])#, RandomErasing()])
    #train_augs = None
    # Dataset and Loader
    dataset_train = TrainDataset(
            args.list_train, args, 
            batch_per_gpu=args.batch_size_per_gpu,
            augmentations=train_augs)
    
    loader_train = data.DataLoader(
        dataset_train,
        batch_size=len(args.gpus),  # we have modified data_parallel
        shuffle=False,  # we do not use this param
        num_workers=int(args.workers),
        drop_last=True,
        pin_memory=False)
    
    print('1 Epoch = {} iters'.format(args.epoch_iters))
    # create loader iterator
    iterator_train = iter(loader_train)

    # load nets into gpu
    if len(args.gpus) > 1:
        segmentation_module = UserScatteredDataParallel(
            segmentation_module,
            device_ids=args.gpus)
        # For sync bn
        patch_replication_callback(segmentation_module)
    segmentation_module.cuda()

    # Set up optimizers
    nets = (net_encoder, net_decoder, crit) if args.unet == False else (unet, crit)
    optimizers = create_optimizers(nets, args)

    # Main loop
    history = {'train': {'epoch': [], 'loss': [], 'acc': []}}

    for epoch in range(args.start_epoch, args.num_epoch + 1):
        train(segmentation_module, iterator_train, optimizers, history, epoch, args)
        # checkpointing
        checkpoint(nets, history, args, epoch)

    print('Training Done!')


if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
        'PyTorch>=0.4.0 is required'

    parser = argparse.ArgumentParser()
    # Model related arguments
    parser.add_argument('--id', default='baseline',
                        help="a name for identifying the model")
    parser.add_argument('--unet', default=True,
                        help="use unet?")
    parser.add_argument('--unet_arch', default='albunet',
                        help="UNet architecture")
    parser.add_argument('--arch_encoder', default='resnet50dilated',
                        help="architecture of net_encoder")
    parser.add_argument('--arch_decoder', default='ppm_deepsup',
                        help="architecture of net_decoder")
    parser.add_argument('--weights_encoder', default='/home/rexma/Desktop/seg/encoder_epoch_20.pth',
                        help="weights to finetune net_encoder")
    parser.add_argument('--weights_decoder', default='',
                        help="weights to finetune net_decoder")
    parser.add_argument('--weights_unet', default='',
                        help="weights to finetune unet")
    parser.add_argument('--fc_dim', default=2048, type=int,
                        help='number of features between encoder and decoder')

    # Path related arguments
    parser.add_argument('--list_train',
                        default='./data/train.odgt')
    parser.add_argument('--list_val',
                        default='./data/validation.odgt')
    parser.add_argument('--root_dataset',
                        default='./data/')

    # optimization related arguments
    parser.add_argument('--gpus', default='0',
                        help='gpus to use, e.g. 0-3 or 0,1,2,3')
    parser.add_argument('--batch_size_per_gpu', default=1, type=int,
                        help='input batch size')
    parser.add_argument('--num_epoch', default=20, type=int,
                        help='epochs to train for')
    parser.add_argument('--start_epoch', default=1, type=int,
                        help='epoch to start training. useful if continue from a checkpoint')
    parser.add_argument('--epoch_iters', default=5000, type=int,
                        help='iterations of each epoch (irrelevant to batch size)')
    parser.add_argument('--optim', default='SGD', help='optimizer')
    parser.add_argument('--lr_encoder', default=0.01, type=float, help='LR')
    parser.add_argument('--lr_decoder', default=0.05, type=float, help='LR')
    parser.add_argument('--lr_pow', default=0.9, type=float,
                        help='power in poly to drop LR')
    parser.add_argument('--beta1', default=0.9, type=float,
                        help='momentum for sgd, beta1 for adam')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weights regularizer')
    parser.add_argument('--deep_sup_scale', default=0.4, type=float,
                        help='the weight of deep supervision loss')
    parser.add_argument('--fix_bn', action='store_true',
                        help='fix bn params')

    # Data related argument
    parser.add_argument('--num_class', default=3, type=int,
                        help='number of classes')
    parser.add_argument('--workers', default=1, type=int,
                        help='number of data loading workers')
    parser.add_argument('--imgSize', default=[127,83,97,130,165,118,142,384,256,528,150,95,140,170],
                        nargs='+', type=int,
                        help='input image size of short edge (int or list)')
    parser.add_argument('--imgMaxSize', default=528, type=int,
                        help='maximum input image size of long edge')
    parser.add_argument('--padding_constant', default=1, type=int,
                        help='maxmimum downsampling rate of the network')
    parser.add_argument('--segm_downsampling_rate', default=1, type=int,
                        help='downsampling rate of the segmentation label')
    parser.add_argument('--random_flip', default=True, type=bool,
                        help='if horizontally flip images when training')

    # Misc arguments
    parser.add_argument('--seed', default=304, type=int, help='manual seed')
    parser.add_argument('--ckpt', default='./ckpt',
                        help='folder to output checkpoints')
    parser.add_argument('--disp_iter', type=int, default=20,
                        help='frequency to display')

    args = parser.parse_args()
    print("Input arguments:")
    for key, val in vars(args).items():
        print("{:16} {}".format(key, val))

    # Parse gpu ids
    all_gpus = parse_devices(args.gpus)
    all_gpus = [x.replace('gpu', '') for x in all_gpus]
    args.gpus = [int(x) for x in all_gpus]
    num_gpus = len(args.gpus)
    args.batch_size = num_gpus * args.batch_size_per_gpu

    args.max_iters = args.epoch_iters * args.num_epoch
    args.running_lr_encoder = args.lr_encoder
    args.running_lr_decoder = args.lr_decoder

    args.arch_encoder = args.arch_encoder.lower()
    args.arch_decoder = args.arch_decoder.lower()

    # Model ID
    if args.unet ==False:
        args.id += '-' + args.arch_encoder
        args.id += '-' + args.arch_decoder
    else:
        args.id += '-' + str(args.unet_arch)

    args.id += '-ngpus' + str(num_gpus)
    args.id += '-batchSize' + str(args.batch_size)
    args.id += '-imgMaxSize' + str(args.imgMaxSize)
    args.id += '-paddingConst' + str(args.padding_constant)
    args.id += '-segmDownsampleRate' + str(args.segm_downsampling_rate)
    
    if args.unet == False:
        args.id += '-LR_encoder' + str(args.lr_encoder)
        args.id += '-LR_decoder' + str(args.lr_decoder)
    else:
        args.id += '-LR_unet' + str(args.lr_encoder)
        
    args.id += '-epoch' + str(args.num_epoch)
    if args.fix_bn:
        args.id += '-fixBN'
    print('Model ID: {}'.format(args.id))

    # FIRST TIME WE TRAINING IT, LOAD THE PRETRAINED WEIGHTS OF ENCODER FROM IMAGENET.
    #args.weights_encoder = '/home/sunjesse/scratch/seg/encoder_epoch_20.pth'

    args.ckpt = os.path.join(args.ckpt, args.id)
    if not os.path.isdir(args.ckpt):
        os.makedirs(args.ckpt)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    main(args)
