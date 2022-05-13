import dataset
from torch.utils.data import DataLoader
from parameters import *
from utils import weights_init
from utils import loss, logger, ramps
from utils import common, metrics
import torch.optim as optim
import IDNet
from tqdm import tqdm
import numpy as np
from collections import OrderedDict
import SimpleITK as sitk


def get_current_consistency_weight(epoch):
    return 1 - 1 * ramps.sigmoid_rampup(epoch, 500)


def update_ema_variables(model, ema_model, global_step=1):
    # Use the true average until the exponential average is more correct
    alpha = 1 * ramps.sigmoid_rampup(global_step, 500)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def train(model, teacher_model, ema_model, TrainMetaData, optimizer, teacher_optimizer, loss_func, n_labels, epoch, train_img_num, test_img_num, threshold):
    print("=======Epoch:{}=======lr:{}".format(epoch, optimizer.state_dict()['param_groups'][0]['lr']))
    model.train()
    train_loss = metrics.LossAverage()

    test_loss = metrics.LossAverage()

    train_dice = metrics.DiceAverage(n_labels)
    test_dice = metrics.DiceAverage(n_labels)

    batImageGenTrains = []
    for data_set in TrainMetaData:
        train_loader = DataLoader(data_set, batch_size=minibatch_size, num_workers=4, shuffle=True)
        batImageGenTrains.append(train_loader)
    iter_test = batImageGenTrains[1].__iter__()
    thresholds = []
    iter_num = np.int(np.floor(test_img_num / minibatch_size))
    for i in tqdm(range(iter_num)):  # img_num
        teacher_model.eval()
        iter_train = batImageGenTrains[0].__iter__()
        for j in range(np.int(np.floor(train_img_num / iter_num))):
            t1_train, t2_train, flair_train, target_train, _ = iter_train.__next__()
            t1_train, t2_train, flair_train, target_train_low = t1_train.float().to(device), t2_train.float().to(device), flair_train.float().to(device), target_train.float().to(device)

            core_1_train, core_2_train, core_3_train, core_4_train, output_train = model(t1_train, t2_train, flair_train)
            _, _, _, _, output_train_test = teacher_model(t1_train, t2_train, flair_train)

            diff = torch.abs(output_train_test.detach() - output_train.detach())
            diff_abs = torch.where(diff > 0.5, 1, 0)
            diff_sum = torch.sum(diff_abs, (1, 2, 3, 4))
            diff_sum = diff_sum.float()
            diff_mean = torch.mean(diff_sum)
            weights = torch.where(diff_sum < threshold, 1.0, get_current_consistency_weight(epoch))
            loss_train = torch.sum(loss_func(output_train, target_train) * weights)
            loss_struct_1_train = torch.sum(loss_func(core_1_train, target_train) * weights)
            loss_struct_2_train = torch.sum(loss_func(core_2_train, target_train) * weights)
            loss_struct_3_train = torch.sum(loss_func(core_3_train, target_train) * weights)
            loss_struct_4_train = torch.sum(loss_func(core_4_train, target_train) * weights)

            loss_train = 16 * loss_train + 1 * (1 * loss_struct_1_train + 2 * loss_struct_2_train + 4 * loss_struct_3_train + 8 * loss_struct_4_train)

            loss = loss_train
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss.update(loss_train.item(), t1_train.size(0))

            train_dice.update(output_train, target_train)
            thresholds.append(diff_mean)

        update_ema_variables(model, ema_model, global_step=epoch)

        for param, teacher_param in zip(ema_model.parameters(), teacher_model.parameters()):
            teacher_param.data.copy_(param.data)

        teacher_model.train()

        t1_test, t2_test, flair_test, target_test, _ = iter_test.__next__()
        t1_test, t2_test, flair_test, target_test = t1_test.float().to(device), \
                                                                    t2_test.float().to(device), \
                                                                    flair_test.float().to(device), \
                                                                    target_test.float().to(device)

        core_1_test, core_2_test, core_3_test, core_4_test, output_test = teacher_model(t1_test, t2_test, flair_test)

        loss_test = torch.sum(loss_func(output_test, target_test))
        loss_struct_1_test = torch.sum(loss_func(core_1_test, target_test))
        loss_struct_2_test = torch.sum(loss_func(core_2_test, target_test))
        loss_struct_3_test = torch.sum(loss_func(core_3_test, target_test))
        loss_struct_4_test = torch.sum(loss_func(core_4_test, target_test))

        loss_test = 16 * loss_test + 1 * (1 * loss_struct_1_test + 2 * loss_struct_2_test + 4 * loss_struct_3_test + 8 * loss_struct_4_test)

        loss = loss_test
        teacher_optimizer.zero_grad()
        loss.backward()
        teacher_optimizer.step()

        test_loss.update(loss_test.item(), t1_test.size(0))
        test_dice.update(output_test, target_test)

    for teacher_param, param in zip(teacher_model.parameters(), model.parameters()):
        param.data = teacher_param.data
    thresholds = torch.tensor(thresholds)
    threshold = torch.mean(thresholds)
    val_log = OrderedDict({'Train_train_Loss': train_loss.avg,
                           'Train_test_Loss': test_loss.avg,
                           'Train_train_dice': train_dice.avg[1],
                           'Train_test_dice': test_dice.avg[1]})
    return val_log, threshold


def val(model, ValidMetaData, loss_func, n_labels, epoch):
    model.eval()
    train_loss = metrics.LossAverage()
    test_loss = metrics.LossAverage()

    train_dice = metrics.DiceAverage(n_labels)
    test_dice = metrics.DiceAverage(n_labels)
    test_HD = metrics.HausdorffDistance()
    test_accuracy = metrics.AccuracyAverage(1)

    batImageGenVals = []

    for data_set in ValidMetaData:
        val_data_loader = DataLoader(data_set, batch_size=1, num_workers=4, shuffle=False)
        batImageGenVals.append(val_data_loader)

    for i in range(len(batImageGenVals)):
        iter_ = batImageGenVals[i].__iter__()

        for ii in range(ValidMetaData[i].img_num):
            t1, t2, flair, target, case_name = iter_.__next__()
            t1, t2, flair, target = t1.float().to(device), t2.float().to(device), flair.float().to(device), target.float().to(device)

            with torch.no_grad():
                _, _, _, _, output = model(t1, t2, flair)
            loss = loss_func(output, target)
            output = torch.where(output > 0.5, 1, 0)
            prediction = sitk.GetImageFromArray(output.cpu().data.numpy()[0, 0, :, :, :])
            result_path = os.path.join(ResDir, str(epoch))
            if not os.path.exists(result_path):
                os.mkdir(result_path)
            sitk.WriteImage(prediction, os.path.join(result_path, case_name[0]))

            if i == 0:
                train_loss.update(loss.item(), t1.size(0))
                train_dice.update(output, target)
            else:
                test_loss.update(loss.item(), t1.size(0))
                test_dice.update(output, target)
                test_accuracy.update(output, target)
                test_HD.update(output, target)

    val_log = OrderedDict({'Val_test_Loss': test_loss.avg, 'Val_train_dice': train_dice.avg[1], 'Val_test_dice': test_dice.avg[1], 'Val_test_accuracy': test_accuracy.avg, 'Val_test_HD': test_HD.avg})
    return val_log


def main():
    trainMetaData = []
    validMetaData = []
    # Training data
    trainMetaData.append(dataset.data_set(DATASET_PATH, split='train', data_type='BraTS'))
    trainMetaData.append(dataset.data_set(DATASET_PATH, split='train', data_type='SISS'))
    # Validation data
    validMetaData.append(dataset.data_set(DATASET_PATH, split='val', data_type='BraTS'))
    validMetaData.append(dataset.data_set(DATASET_PATH, split='val', data_type='SISS'))

    model = IDNet.IDNet().to(device)
    teacher_model = IDNet.IDNet().to(device)
    ema_model = IDNet.IDNet().to(device)

    model.apply(weights_init.init_model)
    for target_param, param in zip(ema_model.parameters(), model.parameters()):
        target_param.data.copy_(param.data)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    teacher_optimizer = optim.Adam(teacher_model.parameters(), lr=lr * 5)
    if multi_GPUs:
        model = torch.nn.DataParallel(model)
        teacher_model = torch.nn.DataParallel(teacher_model)
    Loss_1 = loss.TverskyLoss()
    Loss_2 = loss.structure_loss()
    log = logger.Train_Logger(save_path, "train_log")
    trigger = 0
    threshold = 9999999999.0
    for epoch in range(1, EPOCHS + 1):
        common.adjust_learning_rate(optimizer, epoch, lr)
        common.adjust_learning_rate(teacher_optimizer, epoch, lr * 5)
        train_log, threshold = train(model, teacher_model, ema_model, trainMetaData, optimizer, teacher_optimizer, Loss_2, n_labels, epoch, trainMetaData[0].img_num, trainMetaData[1].img_num, threshold)
        val_log = val(teacher_model, validMetaData, Loss_1, n_labels, epoch)
        log.update(epoch, train_log, val_log)
        state = {'net': teacher_model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
        torch.save(state, os.path.join(save_path, 'latest_model.pth'))
        trigger += 1


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
