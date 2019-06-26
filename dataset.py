import os
import json
import torch
import lib.utils.data as torchdata
import cv2
from torchvision import transforms
import numpy as np
from PIL import Image

class BaseDataset(torchdata.Dataset):
    def __init__(self, odgt, opt, **kwargs):
        # parse options
        self.imgSize = opt.imgSize
        self.imgMaxSize = opt.imgMaxSize
        # max down sampling rate of network to avoid rounding during conv or pooling
        self.padding_constant = opt.padding_constant

        # parse the input list
        self.parse_input_list(odgt, **kwargs)

        # mean and std
        #self.normalize = transforms.Normalize(
        #    mean=[128.0, 128.0, 128.0],
        #    std=[1., 1., 1.])
        
    def parse_input_list(self, odgt, max_sample=-1, start_idx=-1, end_idx=-1):
        if isinstance(odgt, list):
            self.list_sample = odgt
        elif isinstance(odgt, str):
            self.list_sample = [json.loads(x.rstrip()) for x in open(odgt, 'r')]

        if max_sample > 0:
            self.list_sample = self.list_sample[0:max_sample]
        if start_idx >= 0 and end_idx >= 0:     # divide file list
            self.list_sample = self.list_sample[start_idx:end_idx]

        self.num_sample = len(self.list_sample)
        assert self.num_sample > 0
        print('# samples: {}'.format(self.num_sample))

    def img_transform(self, img):
        # image to float
        img = img.astype(np.float32)
        #img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img.copy())/255 #normalize the image between 0 and 1.
        return img

    # Round x to the nearest multiple of p and x' >= x
    def round2nearest_multiple(self, x, p):
        return ((x - 1) // p + 1) * p

    def pad(self, x, y): # Where y = (H, W), the target size.
        if x.shape[0] == y[0] and x.shape[1] == y[1]:
            return x
        diffX = y[1] - x.shape[1]
        diffY = y[0] - x.shape[0]
        
        return cv2.copyMakeBorder(x, diffY // 2, diffY - diffY // 2, diffX // 2, diffX - diffX // 2,
                                    cv2.BORDER_CONSTANT, value=[0,0,0])
        
class TrainDataset(BaseDataset):
    def __init__(self, odgt, opt, batch_per_gpu=1, augmentations=None, **kwargs):
        super(TrainDataset, self).__init__(odgt, opt, **kwargs)
        self.root_dataset = opt.root_dataset
        self.random_flip = opt.random_flip
        # down sampling rate of segm labe
        self.segm_downsampling_rate = opt.segm_downsampling_rate
        self.batch_per_gpu = batch_per_gpu

        # classify images into two classes: 1. h > w and 2. h <= w
        self.batch_record_list = [[], []]

        # override dataset length when trainig with batch_per_gpu > 1
        self.cur_idx = 0
        self.if_shuffled = False

        self.augmentations = augmentations

    def _get_sub_batch(self):
        while True:
            # get a sample record
            this_sample = self.list_sample[self.cur_idx]
            if this_sample['height'] > this_sample['width']:
                self.batch_record_list[0].append(this_sample) # h > w, go to 1st class
            else:
                self.batch_record_list[1].append(this_sample) # h <= w, go to 2nd class

            # update current sample pointer
            self.cur_idx += 1
            if self.cur_idx >= self.num_sample:
                self.cur_idx = 0
                np.random.shuffle(self.list_sample)

            if len(self.batch_record_list[0]) == self.batch_per_gpu:
                batch_records = self.batch_record_list[0]
                self.batch_record_list[0] = []
                break
            elif len(self.batch_record_list[1]) == self.batch_per_gpu:
                batch_records = self.batch_record_list[1]
                self.batch_record_list[1] = []
                break
        return batch_records

    def __getitem__(self, index):
        # NOTE: random shuffle for the first time. shuffle in __init__ is useless
        if not self.if_shuffled:
            np.random.shuffle(self.list_sample)
            self.if_shuffled = True

        # get sub-batch candidates
        batch_records = self._get_sub_batch()

        # resize all images' short edges to the chosen size
        if isinstance(self.imgSize, list):
            this_short_size = np.random.choice(self.imgSize)
        else:
            this_short_size = self.imgSize

        # calculate the BATCH's height and width
        # since we concat more than one samples, the batch's h and w shall be larger than EACH sample
        batch_resized_size = np.zeros((self.batch_per_gpu, 2), np.int32)
        '''
        for i in range(self.batch_per_gpu):
            #Resize images to even H, W
            #if batch_records[i]['height'] % 2 == 1:
            #    batch_records[i]['height']  += 1
            #if batch_records[i]['width'] % 2 == 1:
            #    batch_records[i]['width'] += 1

            img_height, img_width = batch_records[i]['height'], batch_records[i]['width']
            this_scale = min(
                this_short_size / min(img_height, img_width), \
                self.imgMaxSize / max(img_height, img_width))
            img_resized_height, img_resized_width = img_height * this_scale, img_width * this_scale
            batch_resized_size[i, :] = img_resized_height, img_resized_width
        batch_resized_height = np.max(batch_resized_size[:, 0])
        batch_resized_width = np.max(batch_resized_size[:, 1])
    
        # Here we must pad both input image and segmentation map to size h' and w' so that p | h' and p | w'
        batch_resized_height = int(self.round2nearest_multiple(batch_resized_height, self.padding_constant))
        batch_resized_width = int(self.round2nearest_multiple(batch_resized_width, self.padding_constant))
        '''
        assert self.padding_constant >= self.segm_downsampling_rate,\
                'padding constant must be equal or large than segm downsamping rate'
        #print(batch_resized_height)
        #print(batch_resized_width)
        #print()
        d_length = 224
        #batch_images = torch.zeros(self.batch_per_gpu, 3, batch_resized_height, batch_resized_width)
        batch_images = torch.zeros(self.batch_per_gpu, 3, d_length, d_length)
        #batch_segms = torch.zeros(
        #    self.batch_per_gpu, batch_resized_height // self.segm_downsampling_rate, \
        #    batch_resized_width // self.segm_downsampling_rate).long()
        batch_segms = torch.zeros(self.batch_per_gpu, d_length, d_length)

        for i in range(self.batch_per_gpu):
            this_record = batch_records[i]

            # load image and label
            image_path = this_record['fpath_img']
            segm_path = this_record['fpath_segm']
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            segm = cv2.imread(segm_path, cv2.IMREAD_GRAYSCALE)
            
            '''
            cv2.imshow("img", img)
            cv2.imshow("seg", segm)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            '''

            #assert(img.ndim == 3)
            #assert(segm.ndim == 2)
            if img.shape[0] > segm.shape[0]: #FIX MISSING ROW OF PIXELS FOR TRAINING_FULL_GT WHICH IS 384x383 RATHER THAN 384x384
                segm = cv2.copyMakeBorder(segm, 1, 0, 0, 0, cv2.BORDER_CONSTANT, value=[0,0,0])
            #assert(img.shape[0] == segm.shape[0])
            #assert(img.shape[1] == segm.shape[1])
            
            #img = cv2.resize(img, (572, 572))
            #segm

            if self.random_flip is True:
                random_flip = np.random.choice([0, 1])
                if random_flip == 1:
                    img = cv2.flip(img, 1)
                    segm = cv2.flip(segm, 1)
            
            pre_shape = img.shape
        
            ratio = float(d_length)/max(pre_shape)
            new_size = tuple([int(x*ratio) for x in pre_shape])
            
            img = cv2.resize(img, (new_size[1], new_size[0]))
            segm = cv2.resize(segm, (new_size[1], new_size[0]))
            
            #print(img.shape)
            #print(segm.shape)

            img = self.pad(img, (d_length, d_length))
            segm = self.pad(segm, (d_length, d_length))
            
            if self.augmentations is not None:
                img, segm = self.augmentations(img, segm)
            
            img = np.expand_dims(img, 0)
            img = np.concatenate((img, img, img), 0)
            '''
            segm = cv2.resize(
                segm_rounded,
                (segm_rounded.shape[1] // self.segm_downsampling_rate, \
                 segm_rounded.shape[0] // self.segm_downsampling_rate), \
                interpolation=cv2.INTER_NEAREST)
            '''
            img = self.img_transform(img)
            
            batch_images[i][:, :img.shape[1], :img.shape[2]] = img
            batch_segms[i][:segm.shape[0], :segm.shape[1]] = torch.from_numpy(segm.astype(np.int)).long()
        
        batch_segms = ((batch_segms.float() + (batch_segms==255).float())/128.0).float() # label from 0 -> 2, 0 being background/no segmentation class.
        #batch_segms = batch_segms.unsqueeze(1) # NCHW: Channels = number of classes, where each channel is a one hot encoding for class i.
        #batch_segms = torch.cat([(batch_segms==1).float(), (batch_segms==2).float()], 1)
        
        output = dict()
        output['img_data'] = batch_images
        output['seg_label'] = batch_segms.float() 
        return output

    def __len__(self):
        return int(1e10) # It's a fake length due to the trick that every loader maintains its own list
        #return self.num_sampleclass


class ValDataset(BaseDataset):
    def __init__(self, odgt, opt, **kwargs):
        super(ValDataset, self).__init__(odgt, opt, **kwargs)
        #self.root_dataset = opt.root_dataset

    def __getitem__(self, index):
        this_record = self.list_sample[index]
        #print(this_record)
        # load image and label
        image_path = this_record['fpath_img']
        segm_path = this_record['fpath_segm']
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        segm = cv2.imread(segm_path, cv2.IMREAD_GRAYSCALE)

        #ori_height, ori_width, _ = img.shape
        '''
        img_resized_list = []
        for this_short_size in self.imgSize:
            # calculate target height and width
            scale = min(this_short_size / float(min(ori_height, ori_width)),
                        self.imgMaxSize / float(max(ori_height, ori_width)))
            target_height, target_width = int(ori_height * scale), int(ori_width * scale)

            # to avoid rounding in network
            target_height = self.round2nearest_multiple(target_height, self.padding_constant)
            target_width = self.round2nearest_multiple(target_width, self.padding_constant)

            # resize
            img_resized = cv2.resize(img.copy(), (target_width, target_height))
            
            img_resized = np.expand_dims(img, 0)
            img_resized = np.concatenate((img, img, img), 0)
            
            img_resized = img_resized.transpose(2, 0, 1)

            # image transform
            img_resized = self.img_transform(img_resized)

            img_resized = torch.unsqueeze(img_resized, 0)
            img_resized_list.append(img_resized)
        '''
        pre_shape = img.shape
        d_length = 224
        ratio = float(d_length)/max(pre_shape)
        new_size = tuple([int(x*ratio) for x in pre_shape])

        img = cv2.resize(img, (new_size[1], new_size[0]))
        segm = cv2.resize(segm, (new_size[1], new_size[0]))
        img_resized = cv2.resize(img, (new_size[1], new_size[0]))

        img = self.pad(img, (d_length, d_length))
        img_resized = self.pad(img_resized, (d_length, d_length))
        segm = self.pad(segm, (d_length, d_length))
        
        #img_resized = self.img_transform(img_resized)

        img_resized = np.expand_dims(img_resized, 0)
        img_resized = np.concatenate((img_resized, img_resized, img_resized), 0)
        
        img_resized = self.img_transform(img_resized)
        #print(torch.max(img_resized))
        segm = torch.from_numpy(segm.astype(np.int)).long()

        batch_segms = ((segm.float() + (segm==255).float())/128.0).float() #label from 0 -> 2
        #batch_segms = batch_segms.unsqueeze(0)
        
        img_resized = img_resized.unsqueeze(0)
        #print(img_resized.shape)
        #print(batch_segms.shape)

        output = dict()
        output['img_ori'] = img.copy()
        output['img_data'] = img_resized.float()
        output['seg_label'] = batch_segms.float()
        output['info'] = this_record['fpath_img']
        return output

    def __len__(self):
        return self.num_sample


class TestDataset(BaseDataset):
    def __init__(self, odgt, opt, **kwargs):
        super(TestDataset, self).__init__(odgt, opt, **kwargs)

    def __getitem__(self, index):
        this_record = self.list_sample[index]
        # load image and label
        image_path = this_record['fpath_img']
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)

        ori_height, ori_width, _ = img.shape

        img_resized_list = []
        for this_short_size in self.imgSize:
            # calculate target height and width
            scale = min(this_short_size / float(min(ori_height, ori_width)),
                        self.imgMaxSize / float(max(ori_height, ori_width)))
            target_height, target_width = int(ori_height * scale), int(ori_width * scale)

            # to avoid rounding in network
            target_height = self.round2nearest_multiple(target_height, self.padding_constant)
            target_width = self.round2nearest_multiple(target_width, self.padding_constant)

            # resize
            img_resized = cv2.resize(img.copy(), (target_width, target_height))
            
            if img.ndim == 2:
                img = np.expand_dims(img, 0)
                img = np.concatenate((img, img, img), 0)

            # image transform
            img_resized = self.img_transform(img_resized)
            img_resized = torch.unsqueeze(img_resized, 0)
            img_resized_list.append(img_resized)
    
        output = dict()
        output['img_ori'] = img.copy()
        output['img_data'] = [x.contiguous() for x in img_resized_list]
        output['info'] = this_record['fpath_img']
        return output

    def __len__(self):
        return self.num_sample
