import torch
import numpy as np
from utils import experiment_manager, networks, datasets, parsers, metrics, geofiles
from sklearn.metrics import precision_recall_curve, auc
from pathlib import Path


def quantitative_assessment_change(cfg: experiment_manager.CfgNode, run_type: str = 'test'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, *_ = networks.load_checkpoint(cfg.INFERENCE_CHECKPOINT, cfg, device)
    net.eval()

    y_preds = []
    y_trues = []

    with torch.no_grad():
        ds = datasets.MultimodalCDDataset(cfg, run_type, dataset_mode='first_last', no_augmentations=True,
                                          disable_unlabeled=True, disable_multiplier=True)
        for item in ds:
            x_t1 = item['x_t1']
            x_t2 = item['x_t2']
            logits = net(x_t1.unsqueeze(0), x_t2.unsqueeze(0))
            logits= logits[0] if isinstance(logits, tuple) else logits
            y_pred = torch.sigmoid(logits).squeeze().detach().cpu().numpy()
            gt = item['y_change'].squeeze().detach().cpu().numpy()

            y_preds.append(y_pred.flatten())
            y_trues.append(gt.flatten())

    y_preds, y_trues = np.concatenate(y_preds), np.concatenate(y_trues)
    f1 = metrics.f1_score_from_prob(y_preds, y_trues)
    precision = metrics.precsision_from_prob(y_preds, y_trues)
    recall = metrics.recall_from_prob(y_preds, y_trues)

    precisions, recalls, _ = precision_recall_curve(y_trues, y_preds)
    auc_pr = auc(recalls, precisions)

    file = Path(cfg.PATHS.OUTPUT) / 'testing' / 'quantitative_results_change.json'
    if not file.exists():
        data = {}
    else:
        data = geofiles.load_json(file)

    data[cfg.NAME] = {
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'auc': auc_pr,
    }

    geofiles.write_json(file, data)


if __name__ == '__main__':
    args = parsers.deployment_argument_parser().parse_known_args()[0]
    cfg = experiment_manager.setup_cfg(args)
    quantitative_assessment_change(cfg)
