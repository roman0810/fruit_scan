from glob import glob
import torch
from torchvision.io import read_image
from torch.utils.data import Dataset


def biggest_less(sorted_list, target):
    low = 0
    high = len(sorted_list) - 1
    result = -1

    while low <= high:
        mid = (low + high) // 2
        if sorted_list[mid] <= target:
            result = mid
            low = mid + 1
        else:
            high = mid - 1

    return result

class ImageTriplets(Dataset):
    def __init__(self, data_path, triplets=True):
        self.triplets = triplets

        self.class_layer = []
        self.objects_layer = []
        self.classes = []

        self.all_locations = []
        self.total_images = 0

        for cls in glob(f"{data_path}/*"):
            cls_clear = cls.split('/')[-1]
            self.classes.append(cls_clear)

            self.all_locations.append([])
            self.class_layer.append(self.total_images)

            self.objects_layer.append([])
            for obj_path in glob(f"{cls}/*"):
                self.objects_layer[-1].append(self.total_images)
                # выкидываем лишние изображения для делимости каждого объекта на 3
                images = sorted(glob(f"{obj_path}/images/*"))
                images = images[:len(images)-len(images)%3]

                self.all_locations[-1].append(images)

                if self.triplets:
                    self.total_images += len(images)//3
                else:
                    self.total_images += len(images)

            self.DETECTION_CLASSES = len(self.classes)

    def __getitem__(self, index):
        if self.triplets:
            return self._get_triplet(index)
        else:
            return self._get_single(index)

    def __len__(self):
        return self.total_images

    def _get_triplet(self, index):
        cls_index = biggest_less(self.class_layer, index)
        obj_index = biggest_less(self.objects_layer[cls_index], index)

        object_len = len(self.all_locations[cls_index][obj_index])
        pic1 = index - self.objects_layer[cls_index][obj_index]
        pic2 = pic1 + object_len // 3
        pic3 = pic2 + object_len // 3

        image1 = read_image(self.all_locations[cls_index][obj_index][pic1])/255.0
        image2 = read_image(self.all_locations[cls_index][obj_index][pic2])/255.0
        image3 = read_image(self.all_locations[cls_index][obj_index][pic3])/255.0

        target = torch.nn.functional.one_hot(torch.tensor(cls_index), num_classes=self.DETECTION_CLASSES)

        return (image1, image2, image3), target

    def _get_single(self, index):
        cls_index = biggest_less(self.class_layer, index)
        obj_index = biggest_less(self.objects_layer[cls_index], index)
        pic_index = index - self.objects_layer[cls_index][obj_index]

        image = read_image(self.all_locations[cls_index][obj_index][pic_index])/255.0
        target = torch.nn.functional.one_hot(torch.tensor(cls_index), num_classes=self.DETECTION_CLASSES)

        return image, target
