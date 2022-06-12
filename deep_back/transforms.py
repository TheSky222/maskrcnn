import random
from torchvision import transforms as tsf
from torchvision.transforms import functional as F


def _flip_coco_person_keypoints(kps, width):
    flip_inds = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    flipped_data = kps[:, flip_inds]
    flipped_data[..., 0] = width - flipped_data[..., 0]
    # Maintain COCO convention that if visibility == 0, then x, y = 0
    inds = flipped_data[..., 2] == 0
    flipped_data[inds] = 0
    return flipped_data


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
            if "masks" in target:
                target["masks"] = target["masks"].flip(-1)
            if "keypoints" in target:
                keypoints = target["keypoints"]
                keypoints = _flip_coco_person_keypoints(keypoints, width)
                target["keypoints"] = keypoints
        return image, target
    
class Resize(object):
    def __init__(self, shape):
        self.shape = shape
        self.resize_op = tsf.Resize(shape)
        
    def __call__(self, image, target):
        height, width = image.shape[-2:]
        
        scale_h = self.shape[0]/height
        scale_w = self.shape[1]/width
        
        bbox = target['boxes']
        bbox[:, [0, 2]] *= scale_w
        bbox[:, [1, 3]] *= scale_h
        target['boxes'] = bbox
        
        image = self.resize_op(image)
        
        if 'masks' in target:
            masks = target['masks']
            masks = self.resize_op(masks)
            target['masks'] = masks
            #print(height, width, scale_h, scale_w, image.shape, masks.shape)
        
        return image, target
            


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target
