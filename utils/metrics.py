import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from scipy import ndimage
from scipy.spatial.distance import directed_hausdorff

class LossAverage(object):
    """Computes and stores the average and current value for calculate average loss"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = round(self.sum / self.count, 4)
        # print(self.val)

class DiceAverage(object):
    """Computes and stores the average and current value for calculate average loss"""
    def __init__(self,class_num):
        self.class_num = class_num
        self.reset()

    def reset(self):
        self.value = np.asarray([0]*self.class_num, dtype='float64')
        self.avg = np.asarray([0]*self.class_num, dtype='float64')
        self.sum = np.asarray([0]*self.class_num, dtype='float64')
        self.count = 0

    def update(self, logits, targets):
        self.value = DiceAverage.get_dices(logits, targets)
        self.sum += self.value
        self.count += 1
        self.avg = np.around(self.sum / self.count, 4)
        # print(self.value)

    @staticmethod
    def get_dices(logits, targets):
        dices = []
        for class_index in range(targets.size()[1]):
            inter = torch.sum(logits[:, class_index, :, :, :] * targets[:, class_index, :, :, :])
            union = torch.sum(logits[:, class_index, :, :, :]) + torch.sum(targets[:, class_index, :, :, :])
            dice = (2. * inter + 1) / (union + 1)
            dices.append(dice.item())
        return np.asarray(dices)


class AccuracyAverage(object):
    """Computes and stores the average and current value for calculate average loss"""
    def __init__(self, class_num):
        self.class_num = class_num
        self.reset()

    def reset(self):
        self.value = np.asarray([0]*self.class_num, dtype='float64')
        self.avg = np.asarray([0]*self.class_num, dtype='float64')
        self.sum = np.asarray([0]*self.class_num, dtype='float64')
        self.count = 0

    def update(self, logits, targets):
        self.value = self.get_accuracies(logits, targets)
        self.sum += self.value
        self.count += 1
        self.avg = np.around(self.sum / self.count, 4)
        # print(self.value)

    @staticmethod
    def get_accuracies(logits, targets):
        accuracies = []
        for class_index in range(targets.size()[1]):
            inter = torch.sum(logits[:, class_index, :, :, :] * targets[:, class_index, :, :, :])
            union = torch.sum(logits[:, class_index, :, :, :])
            accuracy = inter / (union + 1e-4)
            accuracies.append(accuracy.item())
        return np.asarray(accuracies)


class HausdorffDistance(object):
    """Computes and stores the average and current value for calculate average loss"""
    def __init__(self,):
        self.reset()

    def reset(self):
        self.value = np.asarray([0], dtype='float64')
        self.avg = np.asarray([0], dtype='float64')
        self.sum = np.asarray([0], dtype='float64')
        self.count = 0

    def update(self, logits, targets):
        self.value = self.get_accuracies(logits.cpu().data.numpy(), targets.cpu().data.numpy())
        self.sum += self.value
        self.count += 1
        self.avg = np.around(self.sum / self.count, 4)
        # print(self.value)

    @staticmethod
    def get_accuracies(logits, targets):
        acuracies = []
        for slice in range(targets.shape[2]):
            x = logits[0, 0, slice, :, :]
            y = targets[0, 0, slice, :, :]
            accuracy = directed_hausdorff(x, y)
            acuracies.append(accuracy[0])
        accuracy = np.mean(acuracies)
        return np.asarray(accuracy)
