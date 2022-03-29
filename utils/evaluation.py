import torch
from torch.utils import data as torch_data
import wandb
from utils import datasets, metrics


def model_evaluation_dualtask(net, cfg, device, run_type: str, epoch: float, step: int):

    net.to(device)
    net.eval()

    thresholds = torch.linspace(0.5, 1, 1).to(device)
    measurer_change = metrics.MultiThresholdMetric(thresholds)
    measurer_sem = metrics.MultiThresholdMetric(thresholds)

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
            y_pred_change = torch.sigmoid(logits[0])
            measurer_change.add_sample(gt_change.detach(), y_pred_change.detach())

            # semantics
            logits_stream1_sem_t1, logits_stream1_sem_t2, logits_stream2_sem_t1, logits_stream2_sem_t2 = logits[1:]
            # t1
            gt_sem_t1 = item['y_sem_t1'].to(device)
            y_pred_stream1_sem_t1 = torch.sigmoid(logits_stream1_sem_t1)
            measurer_sem.add_sample(gt_sem_t1.detach(), y_pred_stream1_sem_t1)
            y_pred_stream2_sem_t1 = torch.sigmoid(logits_stream2_sem_t1)
            measurer_sem.add_sample(gt_sem_t1.detach(), y_pred_stream2_sem_t1)
            gt_sem_t2 = item['y_sem_t2'].to(device)
            y_pred_stream1_sem_t2 = torch.sigmoid(logits_stream1_sem_t2)
            measurer_sem.add_sample(gt_sem_t2, y_pred_stream1_sem_t2)
            y_pred_stream2_sem_t2 = torch.sigmoid(logits_stream2_sem_t2)
            measurer_sem.add_sample(gt_sem_t2, y_pred_stream2_sem_t2)

    # change
    f1s_change = measurer_change.compute_f1()
    precisions_change, recalls_change = measurer_change.precision, measurer_change.recall
    f1_change = f1s_change.max().item()
    argmax_f1_change = f1s_change.argmax()
    precision_change = precisions_change[argmax_f1_change].item()
    recall_change = recalls_change[argmax_f1_change].item()

    # semantics
    f1s_sem = measurer_sem.compute_f1()
    precisions_sem, recalls_sem = measurer_sem.precision, measurer_sem.recall
    f1_sem = f1s_sem.max().item()
    argmax_f1_sem = f1s_sem.argmax()
    precision_sem = precisions_sem[argmax_f1_sem].item()
    recall_sem = recalls_sem[argmax_f1_sem].item()

    wandb.log({
        f'{run_type} F1': f1_change,
        f'{run_type} precision': precision_change,
        f'{run_type} recall': recall_change,
        f'{run_type} F1 sem': f1_sem,
        f'{run_type} precision sem': precision_sem,
        f'{run_type} recall sem': recall_sem,
        'step': step, 'epoch': epoch,
    })


def model_evaluation(net, cfg, device, run_type: str, epoch: float, step: int):
    net.to(device)
    net.eval()

    thresholds = torch.linspace(0.5, 1, 1).to(device)
    measurer = metrics.MultiThresholdMetric(thresholds)

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

    f1s = measurer.compute_f1()
    precisions, recalls = measurer.precision, measurer.recall

    f1 = f1s.max().item()
    argmax_f1 = f1s.argmax()
    precision = precisions[argmax_f1].item()
    recall = recalls[argmax_f1].item()

    wandb.log({
        f'{run_type} F1': f1,
        f'{run_type} precision': precision,
        f'{run_type} recall': recall,
        'step': step, 'epoch': epoch,
    })