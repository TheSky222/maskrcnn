# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 20:04:11 2022

@author: wjcy19870122
"""

import os
import torch
import json
import PIL
import glob
import transforms as T
import os.path as osp
import numpy as np
from labelme import utils as labelme_utils

class ChmInstanceSegDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.json_files = glob.glob(osp.join(root, '*.json'))
        
        
    def __getitem__(self, idx):
        json_file = self.json_files[idx]
        image, bboxes, masks, labels = self.__parse_json_file(json_file)
        image = PIL.Image.fromarray(image).convert('RGB')
        num_objs = len(labels)
        
        #convert everything to a torch.tensor
        bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        image_id = torch.tensor([idx])
        area = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        
        target = {}
        target['boxes'] = bboxes
        target['labels'] = labels
        target['masks'] = masks
        target['image_id'] = image_id
        target['area'] = area
        target['iscrowd'] = iscrowd
        
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        
        return image, target
    
    def __len__(self):
        return len(self.json_files)
        
    def __parse_json_file(self, json_file):
        image = None
        masks = []
        bboxes = []
        labels = []
        
        with open(json_file,'r', encoding='UTF-8') as fp:
            json_data = json.load(fp) 
            image = labelme_utils.img_b64_to_arr(json_data['imageData'])
            height, width = image.shape[:2]
            
            for shapes in json_data['shapes']:
                points=shapes['points']
                '''try:
                    label = int(shapes['label'])
                except:'''
                    
                label = 1
                
                mask = self.__polygons_to_mask([height,width], points)
                bbox = self.__mask_to_bbox(mask)
                
                labels.append(label)
                masks.append(mask) 
                bboxes.append(bbox)                
                
            masks = np.array(masks)
            
        return image, bboxes, masks, labels
    
    def __polygons_to_mask(self, img_shape, polygons):
        mask = np.zeros(img_shape, dtype=np.uint8)
        mask = PIL.Image.fromarray(mask)
        xy = list(map(tuple, polygons))
        PIL.ImageDraw.Draw(mask).polygon(xy=xy, outline=1, fill=1)
        mask = np.array(mask, dtype=np.uint8)
        return mask
    
    def __mask_to_bbox(self, mask):
        index = np.argwhere(mask > 0.5)
        rows = index[:, 0]
        clos = index[:, 1]
  
        y1 = np.min(rows)  # y
        x1 = np.min(clos)  # x
   
        y2 = np.max(rows)
        x2 = np.max(clos)
        return [x1, y1, x2, y2]
    
    
#unit tests
if __name__ == '__main__':
    root = 'data/train'
    transform = []
    transform.append(T.ToTensor())
    transform.append(T.Resize((550, 800)))
    transform.append(T.RandomHorizontalFlip(0.5))
    
    dataset = ChmInstanceSegDataset(root, T.Compose(transform))
    print('total found json files: ', len(dataset))
    image, targets = dataset[0]
    
    print(image.shape)