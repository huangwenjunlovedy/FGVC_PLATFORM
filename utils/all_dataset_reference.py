# -*- coding: utf-8 -*-
"""
@author:yangxb23
@github:https://github.com/xuebin-yang/
@data:02/12/2021
"""

import torch
import torchvision
import numpy as np

import time
from utils.load_pretrain_model import *
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PyQt5 import QtCore, QtGui, QtWidgets
from time import sleep

import cgitb
cgitb.enable(format='text')   # output the error message

best_acc1 = 0
best_acc5 = 0


def reference(data_path, dataset_name, widget_name=None, line_edit=None, func=None):  # widget_name=textbrowser
    input_size = 448

    resolution_edit = "{} x {}".format(input_size, input_size)  # 没办法实时显示分辨率
    line_edit.setText(resolution_edit)

    global best_acc1
    global best_acc5
    if torch.cuda.is_available():
        # th_emit_signal.emit("Use GPU for reference!!!")
        func("Use GPU for reference!!!")
    else:
        # th_emit_signal.emit("Using cpu for reference!!!")
        func("Using cpu for reference!!!")

    # th_emit_signal.emit("=> creating model: ResNet50!!!")
    func("=> creating model: ResNet50!!!")
    # change output_features
    model = load_model(dataset_name)
    # th_emit_signal.emit("loaded pretrained model!!!")
    # th_emit_signal.emit("=> {} reference!!!".format(dataset_name))
    func("loaded pretrained model!!!")
    func("=> {} reference!!!".format(dataset_name))
    model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    cudnn.benchmark = True

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if input_size == 224:
        # th_emit_signal.emit('Finetune resolution: 224 x 224')
        func('Finetune resolution: 224 x 224')
        val_transforms = transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            normalize
        ])
    elif input_size == 448:
        # th_emit_signal.emit('Finetune resolution: 448 x 448')
        func('Finetune resolution: 448 x 448')
        val_transforms = transforms.Compose([
            transforms.Resize(size=600),
            transforms.CenterCrop(size=448),
            transforms.ToTensor(),
            normalize
        ])

    val_dataset = datasets.ImageFolder(root=data_path, transform=val_transforms)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)

    # evaluate on validation set
    reference_start = time.time()
    QtWidgets.QApplication.processEvents()
    acc1, acc5 = validate(val_loader, model, criterion, widget_name, func)
    reference_end = time.time()
    # th_emit_signal.emit("acc1: " + str(acc1.item()))
    # th_emit_signal.emit("acc5: " + str(acc5.item()))
    # th_emit_signal.emit('total referene time elapses {:6.2f} s'.format((reference_end - reference_start)))
    func("acc1: " + str(acc1.item()))
    func("acc5: " + str(acc5.item()))
    func('total referene time elapses {:6.2f} s'.format((reference_end - reference_start)))
    return format(acc1, '.2f'), format(acc5, '.2f'), format(reference_end - reference_start, '.0f')


def validate(val_loader, model, criterion, widget_name, func=None):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader), [losses, top1, top5], prefix='Test: ', textbrowser=widget_name, func=func)
    QtWidgets.QApplication.processEvents()

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            QtWidgets.QApplication.processEvents()
            if torch.cuda.is_available():
                images = images.cuda()
                target = target.cuda()

            # compute output
            QtWidgets.QApplication.processEvents()
            output = model(images)
            QtWidgets.QApplication.processEvents()
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % 10 == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        func(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
    return top1.avg, top5.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix="", textbrowser=None, func=None):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.printf = func
        self.textb = textbrowser

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        self.printf('   '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1, 5)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res