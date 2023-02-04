import torch
from utils import experiment_manager, networks, datasets, parsers, geofiles, evaluation
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def quantitative_assessment_semantic(cfg: experiment_manager.CfgNode, run_type: str = 'test'):
    net, *_ = networks.load_checkpoint(cfg, device)
    net.eval()

    measurer = evaluation.Measurer(run_type, 'sem')

    ds = datasets.MultimodalCDDataset(cfg, run_type, dataset_mode='first_last', no_augmentations=True,
                                      disable_unlabeled=True, disable_multiplier=True)

    with torch.no_grad():
        for item in ds:
            # semantic predictions
            x_t1, x_t2 = item['x_t1'].to(device), item['x_t2'].to(device)
            logits = net(x_t1.unsqueeze(0), x_t2.unsqueeze(0))
            if cfg.MODEL.TYPE == 'dtsiameseunet':
                _, logits_t1, logits_t2 = logits
            else:
                logits_t1, logits_t2 = logits[5:]
            y_hat_t1, y_hat_t2 = torch.sigmoid(logits_t1), torch.sigmoid(logits_t2)

            # semantic labels
            y_t1, y_t2 = item['y_sem_t1'].to(device), item['y_sem_t2'].to(device)

            measurer.add_sample(y_t1, y_hat_t1)
            measurer.add_sample(y_t2, y_hat_t2)

    file = Path(cfg.PATHS.OUTPUT) / 'testing' / f'quantitative_results_semantic_{run_type}.json'
    if not file.exists():
        data = {}
    else:
        data = geofiles.load_json(file)

    data[cfg.NAME] = {
        'f1_score': measurer.f1().item(),
        'precision': measurer.precision().item(),
        'recall': measurer.recall().item(),
        'iou': measurer.iou().item(),
    }

    geofiles.write_json(file, data)


if __name__ == '__main__':
    args = parsers.deployment_argument_parser().parse_known_args()[0]
    cfg = experiment_manager.setup_cfg(args)
    quantitative_assessment_semantic(cfg, run_type='test')
