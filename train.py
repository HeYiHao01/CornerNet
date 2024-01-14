import datetime
import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.cornernet import CornerNet_Resnet50, CornerNet_MobilenetV3
from nets.training_utils import get_lr_scheduler, set_optimizer_lr
from utils.loss_history import LossHistory
from utils.dataloader import CornernetDataset, corner_dataset_collate
from utils.utils import download_weights, get_classes, show_config
from utils.utils_fit import fit_one_epoch


if __name__ == "__main__":
    Cuda = True
    #   fp16        是否使用混合精度训练, 可减少约一半的显存
    fp16 = False
    classes_path = 'config/classes_path.txt'
    model_path = 'logs/saved/mobilenet_v3_v4_SA/best_val_weights.pth'  # mobilenetv3
    input_shape = [640, 640]

    # backbone = "resnet50"
    backbone = "mobilenetv3"
    pretrained = False

    # train on two stages
    Init_Epoch = 0
    Freeze_Epoch = 100
    Freeze_batch_size = 16

    UnFreeze_Epoch = 200
    Unfreeze_batch_size = 16

    Freeze_Train = True

    Init_lr = 1e-4
    Min_lr = Init_lr * 0.01

    optimizer_type = "adam"
    momentum = 0.9
    weight_decay = 5e-4

    lr_decay_type = 'cos'

    # save weights per save_period epoch
    save_period = 1e3
    # save log dir
    save_dir = 'logs'

    num_workers = 4

    train_annotation_path = 'config/total_v3_v4.txt'
    val_annotation_path = 'config/total_v3.txt'

    device = torch.device('cuda' if torch.cuda.is_available() and Cuda else 'cpu')
    local_rank = 0

    if pretrained:
        download_weights(backbone)

    class_names, num_classes = get_classes(classes_path)

    if backbone == "resnet50":
        model = CornerNet_Resnet50(num_classes, pretrained=pretrained)
    elif backbone == 'mobilenetv3':
        model = CornerNet_MobilenetV3(num_classes, pretrained)

    if model_path != '':
        if local_rank == 0:
            print('Load weights {}.'.format(model_path))

        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)

    # record loss
    if local_rank == 0:
        time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
        log_dir = os.path.join(save_dir, "loss_" + str(time_str))
        loss_history = LossHistory(log_dir, model, input_shape=input_shape)
    else:
        loss_history = None

    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler

        scaler = GradScaler()
    else:
        scaler = None

    model_train = model.train()

    if Cuda:
        model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_train = model_train.cuda()

    with open(train_annotation_path) as f:
        train_lines = f.readlines()
    with open(val_annotation_path) as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)

    if local_rank == 0:
        show_config(
            classes_path=classes_path, model_path=model_path, input_shape=input_shape,
            Init_Epoch=Init_Epoch, Freeze_Epoch=Freeze_Epoch, UnFreeze_Epoch=UnFreeze_Epoch,
            Freeze_batch_size=Freeze_batch_size, Unfreeze_batch_size=Unfreeze_batch_size, Freeze_Train=Freeze_Train,
            Init_lr=Init_lr, Min_lr=Min_lr, optimizer_type=optimizer_type, momentum=momentum,
            lr_decay_type=lr_decay_type,
            save_period=save_period, save_dir=save_dir, num_workers=num_workers, num_train=num_train, num_val=num_val
        )

        wanted_step = 5e4 if optimizer_type == "sgd" else 1.5e4
        total_step = num_train // Unfreeze_batch_size * UnFreeze_Epoch

    if True:
        UnFreeze_flag = False
        if Freeze_Train:
            model.freeze_backbone()

        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

        nbs = 64
        lr_limit_max = 5e-4 if optimizer_type == 'adam' else 5e-2
        lr_limit_min = 2.5e-4 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        optimizer = {
            'adam': optim.Adam(model.parameters(), Init_lr_fit, betas=(momentum, 0.999), weight_decay=weight_decay),
            'sgd': optim.SGD(model.parameters(), Init_lr_fit, momentum=momentum, nesterov=True,
                             weight_decay=weight_decay)
        }[optimizer_type]

        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，请扩充数据集")

        train_dataset = CornernetDataset(train_lines, input_shape, num_classes, train=True)
        val_dataset = CornernetDataset(val_lines, input_shape, num_classes, train=False)

        train_sampler = None
        val_sampler = None
        shuffle = True

        gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                         pin_memory=True,
                         drop_last=True, collate_fn=corner_dataset_collate, sampler=train_sampler)
        gen_val = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                             pin_memory=True,
                             drop_last=True, collate_fn=corner_dataset_collate, sampler=val_sampler)

        for epoch in range(Init_Epoch, UnFreeze_Epoch):
            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                batch_size = Unfreeze_batch_size

                nbs = 64
                lr_limit_max = 5e-4 if optimizer_type == 'adam' else 5e-2
                lr_limit_min = 2.5e-4 if optimizer_type == 'adam' else 5e-4
                Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
                Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

                model.unfreeze_backbone()

                epoch_step = num_train // batch_size
                epoch_step_val = num_val // batch_size

                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("数据集过小，请扩充数据集")

                gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                                 pin_memory=True,
                                 drop_last=True, collate_fn=corner_dataset_collate, sampler=train_sampler)
                gen_val = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                                     pin_memory=True,
                                     drop_last=True, collate_fn=corner_dataset_collate, sampler=val_sampler)

                UnFreeze_flag = True

            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            fit_one_epoch(model_train, model, loss_history, optimizer, epoch,
                          epoch_step, epoch_step_val, gen, gen_val, UnFreeze_Epoch, Cuda, fp16, scaler, backbone,
                          save_period, save_dir, local_rank)

        if local_rank == 0:
            loss_history.writer.close()
