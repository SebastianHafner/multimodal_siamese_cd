import torch
from pathlib import Path
import numpy as np
from utils import experiment_manager, networks, datasets, parsers, geofiles


T1 = '2022p1'
T2 = '2022p2'
SITE = 'stockholm'
PATCH_SIZE = 256


def inference_change(cfg: experiment_manager.CfgNode):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, *_ = networks.load_checkpoint(cfg.INFERENCE_CHECKPOINT, cfg, device)
    net.eval()

    pred_folder = Path(cfg.PATHS.OUTPUT) / 'inference' / cfg.NAME
    pred_folder.mkdir(exist_ok=True)

    ds = datasets.SimpleInferenceDataset(cfg, site=SITE, t1=T1, t2=T2, patch_size=PATCH_SIZE)
    y_pred_change = ds.get_arr()
    transform, crs = ds.get_geo()

    with torch.no_grad():
        for index in range(len(ds)):
            item = ds.__getitem__(index)
            x_t1 = item['x_t1']
            x_t2 = item['x_t2']
            logits = net(x_t1.unsqueeze(0), x_t2.unsqueeze(0))
            logits_change = logits[0] if isinstance(logits, tuple) else logits
            y_pred_change_patch = torch.sigmoid(logits_change).squeeze().detach().cpu().numpy()

            i_min, i_max = item['i_min'], item['i_max']
            j_min, j_max = item['j_min'], item['j_max']

            y_pred_change[i_min:i_max, j_min:j_max, 0] = (y_pred_change_patch * 100).astype(np.uint8)

    pred_file = pred_folder / f'pred_change_{cfg.NAME}_{SITE}_{T1}_{T2}.tif'
    geofiles.write_tif(pred_file, y_pred_change, transform, crs)


if __name__ == '__main__':
    args = parsers.deployment_argument_parser().parse_known_args()[0]
    cfg = experiment_manager.setup_cfg(args)
    inference_change(cfg)