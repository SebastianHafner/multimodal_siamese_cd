import torch
from torch.utils import data as torch_data
import wandb
from utils import datasets

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Measurer(object):
    def __init__(self, name: str = None, threshold: float = 0.5):

        self.name = name
        self.threshold = threshold

        self.TP = 0
        self.TN = 0
        self.FP = 0
        self.FN = 0

        self._precision = None
        self._recall = None

        self.eps = 10e-05

    def add_sample(self, y: torch.Tensor, y_hat: torch.Tensor):
        y = y.bool()
        y_hat = y_hat > self.threshold

        self.TP += torch.sum(y & y_hat).float()
        self.TN += torch.sum(~y & ~y_hat).float()
        self.FP += torch.sum(y_hat & ~y).float()
        self.FN += torch.sum(~y_hat & y).float()

    def precision(self):
        if self._precision is None:
            self._precision = self.TP / (self.TP + self.FP + self.eps)
        return self._precision

    def recall(self):
        if self._recall is None:
            self._recall = self.TP / (self.TP + self.FN + self.eps)
        return self._recall

    def compute_basic_metrics(self):
        false_pos_rate = self.FP / (self.FP + self.TN + self.eps)
        false_neg_rate = self.FN / (self.FN + self.TP + self.eps)
        return false_pos_rate, false_neg_rate

    def f1(self):
        return (2 * self.precision() * self.recall()) / (self.precision() + self.recall() + self.eps)

    def iou(self):
        return self.TP / (self.TP + self.FP + self.FN + self.eps)

    def oa(self):
        return (self.TP + self.TN) / (self.TP + self.TN + self.FP + self.FN + self.eps)

    def is_empty(self):
        return True if (self.TP + self.TN + self.FP + self.FN) == 0 else False


def model_evaluation(net, cfg, run_type: str, epoch: float, step: int, early_stopping: bool = False):
    net.to(device)
    net.eval()

    measurer = Measurer('change')

    ds = datasets.MultimodalCDDataset(cfg, run_type, no_augmentations=True, dataset_mode='first_last',
                                      disable_multiplier=True, disable_unlabeled=True)
    dataloader = torch_data.DataLoader(ds, batch_size=1, num_workers=0, shuffle=False, drop_last=False)
    with torch.no_grad():
        for step, item in enumerate(dataloader):
            x_t1 = item['x_t1'].to(device)
            x_t2 = item['x_t2'].to(device)
            logits = net(x_t1, x_t2)
            y_pred = torch.sigmoid(logits)

            gt = item['y_change'].to(device)
            measurer.add_sample(gt.detach(), y_pred.detach())

    return_value = None
    if not measurer.is_empty():
        f1 = measurer.f1()
        false_pos_rate, false_neg_rate = measurer.compute_basic_metrics()

        suffix = 'earlystopping ' if early_stopping else ''
        wandb.log({
            suffix + f'{run_type} {measurer.name} F1': measurer.f1(),
            suffix + f'{run_type} {measurer.name} fpr': false_pos_rate,
            suffix + f'{run_type} {measurer.name} fnr': false_neg_rate,
            'step': step, 'epoch': epoch,
        })

        if measurer.name == 'all':
            return_value = f1

    return return_value


def model_evaluation_dt(net, cfg, run_type: str, epoch: float, step: int, early_stopping: bool = False):
    net.to(device)
    net.eval()

    measurer_change = Measurer('change')
    measurer_sem = Measurer('sem')

    ds = datasets.MultimodalCDDataset(cfg, run_type, no_augmentations=True, dataset_mode='first_last',
                                      disable_multiplier=True, disable_unlabeled=True)
    dataloader = torch_data.DataLoader(ds, batch_size=1, num_workers=0, shuffle=False, drop_last=False)
    with torch.no_grad():
        for step, item in enumerate(dataloader):
            x_t1 = item['x_t1'].to(device)
            x_t2 = item['x_t2'].to(device)
            logits_change, logits_sem_t1, logits_sem_t2 = net(x_t1, x_t2)

            # change
            gt_change = item['y_change'].to(device)
            y_pred_change = torch.sigmoid(logits_change)
            measurer_change.add_sample(gt_change.detach(), y_pred_change.detach())

            # semantics
            # t1
            gt_sem_t1 = item['y_sem_t1'].to(device)
            y_pred_sem_t1 = torch.sigmoid(logits_sem_t1)
            measurer_sem.add_sample(gt_sem_t1.detach(), y_pred_sem_t1)
            # t2
            gt_sem_t2 = item['y_sem_t2'].to(device)
            y_pred_sem_t2 = torch.sigmoid(logits_sem_t2)
            measurer_sem.add_sample(gt_sem_t2, y_pred_sem_t2)

    return_value = None
    for measurer in (measurer_change, measurer_sem):
        if not measurer.is_empty():
            f1 = measurer.f1()
            false_pos_rate, false_neg_rate = measurer.compute_basic_metrics()

            suffix = 'earlystopping ' if early_stopping else ''
            wandb.log({
                suffix + f'{run_type} {measurer.name} F1': measurer.f1(),
                suffix + f'{run_type} {measurer.name} fpr': false_pos_rate,
                suffix + f'{run_type} {measurer.name} fnr': false_neg_rate,
                'step': step, 'epoch': epoch,
            })

            if measurer.name == 'all':
                return_value = f1

    return return_value


def model_evaluation_mm_dt(net, cfg, run_type: str, epoch: float, step: int, early_stopping: bool = False):
    net.to(device)
    net.eval()

    measurer_change = Measurer('change')
    measurer_sem = Measurer('sem')

    ds = datasets.MultimodalCDDataset(cfg, run_type, no_augmentations=True, dataset_mode='first_last',
                                      disable_multiplier=True, disable_unlabeled=True)
    dataloader = torch_data.DataLoader(ds, batch_size=1, num_workers=0, shuffle=False, drop_last=False)

    with torch.no_grad():
        for step, item in enumerate(dataloader):
            x_t1 = item['x_t1'].to(device)
            x_t2 = item['x_t2'].to(device)
            logits = net(x_t1, x_t2)

            # change
            gt_change = item['y_change'].to(device)
            y_pred_change = torch.sigmoid(logits[0]).detach()
            measurer_change.add_sample(gt_change, y_pred_change)

            # semantics
            logits_fusion_sem_t1, logits_fusion_sem_t2 = logits[5:]
            # t1
            gt_sem_t1 = item['y_sem_t1'].to(device)
            y_pred_fusion_sem_t1 = torch.sigmoid(logits_fusion_sem_t1).detach()
            measurer_sem.add_sample(gt_sem_t1, y_pred_fusion_sem_t1)
            # t2
            gt_sem_t2 = item['y_sem_t2'].to(device)
            y_pred_fusion_sem_t2 = torch.sigmoid(logits_fusion_sem_t2).detach()
            measurer_sem.add_sample(gt_sem_t2, y_pred_fusion_sem_t2)

    return_value = None
    for measurer in (measurer_change, measurer_sem):
        if not measurer.is_empty():
            f1 = measurer.f1()
            false_pos_rate, false_neg_rate = measurer.compute_basic_metrics()

            suffix = 'earlystopping ' if early_stopping else ''
            wandb.log({
                suffix + f'{run_type} {measurer.name} F1': measurer.f1(),
                suffix + f'{run_type} {measurer.name} fpr': false_pos_rate,
                suffix + f'{run_type} {measurer.name} fnr': false_neg_rate,
                'step': step, 'epoch': epoch,
            })

            if measurer.name == 'all':
                return_value = f1

    return return_value
