import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from utils import experiment_manager, networks, datasets, metrics, geofiles, parsers
from sklearn.metrics import precision_recall_curve, auc


def quantitative_assessment_semantic(cfg: experiment_manager.CfgNode, run_type: str = 'test'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, *_ = networks.load_checkpoint(cfg.INFERENCE_CHECKPOINT, cfg, device)
    net.eval()
    ds = datasets.MultimodalCDDataset(cfg, run_type, dataset_mode='first_last', no_augmentations=True,
                                      disable_unlabeled=True, disable_multiplier=True)

    y_trues, y_preds_s1, y_preds_s2 = [], [], []

    with torch.no_grad():
        for item in tqdm(ds):
            # semantic labels
            gt_t1, gt_t2 = item['y_sem_t1'].to(device), item['y_sem_t2'].to(device)
            y_trues.extend([gt_t1.flatten(), gt_t2.flatten()])

            # semantic predictions
            x_t1, x_t2 = item['x_t1'].to(device), item['x_t2'].to(device)
            logits = net(x_t1.unsqueeze(0), x_t2.unsqueeze(0))
            if cfg.MODEL.TYPE == 'dtsiameseunet':
                _, logits_t1, logits_t2 = logits
                pred_t1, pred_t2 = torch.sigmoid(logits_t1), torch.sigmoid(logits_t2)
                if cfg.DATALOADER.INPUT_MODE == 's1':
                    y_preds_s1.extend([pred_t1.flatten(), pred_t2.flatten()])
                else:
                    y_preds_s2.extend([pred_t1.flatten(), pred_t2.flatten()])
            else:
                logits_s1_t1, logits_s1_t2 = logits[1:3]
                pred_s1_t1, pred_s1_t2 = torch.sigmoid(logits_s1_t1), torch.sigmoid(logits_s1_t2)
                logits_s2_t1, logits_s2_t2 = logits[3:]
                pred_s2_t1, pred_s2_t2 = torch.sigmoid(logits_s2_t1), torch.sigmoid(logits_s2_t2)
                y_preds_s1.extend([pred_s1_t1.flatten(), pred_s1_t2.flatten()])
                y_preds_s2.extend([pred_s2_t1.flatten(), pred_s2_t2.flatten()])

    y_preds_s1 = torch.cat(y_preds_s1).flatten().cpu().numpy() if len(y_preds_s1) > 0 else None
    y_preds_s2 = torch.cat(y_preds_s2).flatten().cpu().numpy() if len(y_preds_s2) > 0 else None
    y_trues = torch.cat(y_trues).flatten().cpu().numpy()

    file = Path(cfg.PATHS.OUTPUT) / 'testing' / f'quantitative_results_semantic_{run_type}.json'
    if not file.exists():
        data = {}
    else:
        data = geofiles.load_json(file)
    data[cfg.NAME] = {}

    for sensor, y_preds in zip(['s1', 's2'], [y_preds_s1, y_preds_s2]):
        if y_preds is None:
            f1 = precision = recall = auc_pr = None
        else:
            f1 = metrics.f1_score_from_prob(y_preds, y_trues)
            precision = metrics.precsision_from_prob(y_preds, y_trues)
            recall = metrics.recall_from_prob(y_preds, y_trues)
            precisions, recalls, _ = precision_recall_curve(y_trues, y_preds)
            auc_pr = auc(recalls, precisions)

        data[cfg.NAME][sensor] = {
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'auc': auc_pr,
        }

    geofiles.write_json(file, data)


if __name__ == '__main__':
    args = parsers.deployment_argument_parser().parse_known_args()[0]
    cfg = experiment_manager.setup_cfg(args)
    quantitative_assessment_semantic(cfg, run_type='validation')
