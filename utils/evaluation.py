import torch
from torch.utils import data as torch_data
import wandb
from utils import datasets, metrics


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