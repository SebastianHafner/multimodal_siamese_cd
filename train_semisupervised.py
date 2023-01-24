import sys
import os
import timeit

import torch
from torch import optim
from torch.utils import data as torch_data

from tabulate import tabulate
import wandb
import numpy as np

from utils import networks, datasets, loss_functions, evaluation, experiment_manager, parsers


def run_training(cfg):
    run_config = {
        'CONFIG_NAME': cfg.NAME,
        'device': device,
        'epochs': cfg.TRAINER.EPOCHS,
        'learning rate': cfg.TRAINER.LR,
        'batch size': cfg.TRAINER.BATCH_SIZE,
    }
    table = {'run config name': run_config.keys(),
             ' ': run_config.values(),
             }
    print(tabulate(table, headers='keys', tablefmt="fancy_grid", ))

    net = networks.create_network(cfg)
    net.to(device)
    optimizer = optim.AdamW(net.parameters(), lr=cfg.TRAINER.LR, weight_decay=0.01)

    sup_criterion = loss_functions.get_criterion(cfg.MODEL.LOSS_TYPE)
    cons_criterion = loss_functions.get_criterion(cfg.CONSISTENCY_TRAINER.LOSS_TYPE)

    # reset the generators
    dataset = datasets.MultimodalCDDataset(cfg=cfg, run_type='training')
    print(dataset)

    dataloader_kwargs = {
        'batch_size': cfg.TRAINER.BATCH_SIZE,
        'num_workers': 0 if cfg.DEBUG else cfg.DATALOADER.NUM_WORKER,
        'shuffle': cfg.DATALOADER.SHUFFLE,
        'drop_last': True,
        'pin_memory': True,
    }
    dataloader = torch_data.DataLoader(dataset, **dataloader_kwargs)

    # unpacking cfg
    epochs = cfg.TRAINER.EPOCHS
    save_checkpoints = cfg.SAVE_CHECKPOINTS
    steps_per_epoch = len(dataloader)

    # tracking variables
    global_step = epoch_float = 0

    for epoch in range(1, epochs + 1):
        print(f'Starting epoch {epoch}/{epochs}.')

        start = timeit.default_timer()
        fusion_loss_set, stream1_loss_set, stream2_loss_set = [], [], []
        loss_set, sup_loss_set, cons_loss_set = [], [], []

        n_labeled, n_notlabeled = 0, 0

        for i, batch in enumerate(dataloader):

            net.train()
            optimizer.zero_grad()

            x_t1 = batch['x_t1'].to(device)
            x_t2 = batch['x_t2'].to(device)

            logits_fusion, logits_stream1, logits_stream2 = net(x_t1, x_t2)
            y_pred_stream1 = torch.sigmoid(logits_stream1)
            y_pred_stream2 = torch.sigmoid(logits_stream2)

            sup_loss, cons_loss = None, None

            is_labeled = batch['is_labeled']
            n_labeled += torch.sum(is_labeled).item()
            if is_labeled.any():
                # change detection
                gt_change = batch['y_change'].to(device)
                fusion_loss = sup_criterion(logits_fusion[is_labeled, ], gt_change[is_labeled, ])
                stream1_loss = sup_criterion(logits_stream1[is_labeled,], gt_change[is_labeled,])
                stream2_loss = sup_criterion(logits_stream2[is_labeled,], gt_change[is_labeled,])

                sup_loss = (fusion_loss + stream1_loss + stream2_loss) / 3
                sup_loss = cfg.CONSISTENCY_TRAINER.LOSS_FACTOR * sup_loss

                sup_loss_set.append(sup_loss.item())
                fusion_loss_set.append(fusion_loss.item())
                stream1_loss_set.append(stream1_loss.item())
                stream2_loss_set.append(stream2_loss.item())

            if not is_labeled.all():
                is_not_labeled = torch.logical_not(is_labeled)
                n_notlabeled += torch.sum(is_not_labeled).item()

                if cfg.CONSISTENCY_TRAINER.LOSS_TYPE == 'L2':
                    cons_loss = cons_criterion(y_pred_stream1[is_not_labeled,], y_pred_stream2[is_not_labeled,])
                else:
                    cons_loss = cons_criterion(logits_stream1[is_not_labeled,], y_pred_stream2[is_not_labeled,])
                cons_loss = (1 - cfg.CONSISTENCY_TRAINER.LOSS_FACTOR) * cons_loss
                cons_loss_set.append(cons_loss.item())

            if sup_loss is None and cons_loss is not None:
                loss = cons_loss
            elif sup_loss is not None and cons_loss is not None:
                loss = sup_loss + cons_loss
            else:
                loss = sup_loss

            loss.backward()
            optimizer.step()

            loss_set.append(loss.item())

            global_step += 1
            epoch_float = global_step / steps_per_epoch

            if cfg.DEBUG:
                break

            if global_step % cfg.LOG_FREQ == 0:
                print(f'Logging step {global_step} (epoch {epoch_float:.2f}).')

                # evaluation on sample of training and validation set
                evaluation.model_evaluation(net, cfg, device, 'training', epoch_float, global_step)
                evaluation.model_evaluation(net, cfg, device, 'validation', epoch_float, global_step)

                # logging
                time = timeit.default_timer() - start
                wandb.log({
                    'fusion_loss': np.mean(fusion_loss_set) if len(fusion_loss_set) > 0 else 0,
                    'stream1_loss': np.mean(stream1_loss_set) if len(stream1_loss_set) > 0 else 0,
                    'stream2_loss': np.mean(stream2_loss_set) if len(stream2_loss_set) > 0 else 0,
                    'sup_loss': np.mean(sup_loss_set) if len(sup_loss_set) > 0 else 0,
                    'cons_loss': np.mean(cons_loss_set) if len(cons_loss_set) > 0 else 0,
                    'loss': np.mean(loss_set),
                    'labeled_percentage': n_labeled / (n_labeled + n_notlabeled) * 100,
                    'time': time,
                    'step': global_step,
                    'epoch': epoch_float,
                })
                start = timeit.default_timer()
                n_labeled, n_notlabeled = 0, 0
                fusion_loss_set, stream1_loss_set, stream2_loss_set = [], [], []
                loss_set, sup_loss_set, cons_loss_set = [], [], []
            # end of batch

        if not cfg.DEBUG:
            assert (epoch == epoch_float)
        print(f'epoch float {epoch_float} (step {global_step}) - epoch {epoch}')
        # evaluation at the end of an epoch
        evaluation.model_evaluation(net, cfg, device, 'training', epoch_float, global_step)
        evaluation.model_evaluation(net, cfg, device, 'validation', epoch_float, global_step)
        evaluation.model_evaluation(net, cfg, device, 'test', epoch_float, global_step)

        if epoch in save_checkpoints:
            print(f'saving network', flush=True)
            networks.save_checkpoint(net, optimizer, epoch, global_step, cfg)


if __name__ == '__main__':
    args = parsers.default_argument_parser().parse_known_args()[0]
    cfg = experiment_manager.setup_cfg(args)

    # make training deterministic
    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('=== Runnning on device: p', device)

    wandb.init(
        name=cfg.NAME,
        config=cfg,
        entity='population_mapping',
        project=args.project,
        tags=['ssl', 'cd', 'siamese', 'spacenet7', ],
        mode='online' if not cfg.DEBUG else 'disabled',
    )

    try:
        run_training(cfg)
    except KeyboardInterrupt:
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
