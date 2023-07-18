import sys
import os
import timeit

import torch
from torch import optim
from torch.utils import data as torch_data

import wandb
import numpy as np

from utils import networks, datasets, loss_functions, evaluation, experiment_manager, parsers, helpers
import random


def run_training(cfg):
    net = networks.create_network(cfg)
    net.to(device)
    optimizer = optim.AdamW(net.parameters(), lr=cfg.TRAINER.LR, weight_decay=0.01)

    sup_criterion = loss_functions.get_criterion(cfg.MODEL.LOSS_TYPE)
    cons_criterion = loss_functions.get_criterion(cfg.CONSISTENCY_TRAINER.LOSS_TYPE)

    # reset the generators
    labeled_dataset = datasets.MultimodalCDDataset(cfg=cfg, run_type='train', disable_unlabeled=True)
    unlabeled_dataset = datasets.MultimodalCDDataset(cfg=cfg, run_type='train', only_unlabeled=True)
    print(labeled_dataset, unlabeled_dataset)

    dataloader_kwargs = {
        'batch_size': int(cfg.TRAINER.BATCH_SIZE // 2),
        'num_workers': 0 if cfg.DEBUG else cfg.DATALOADER.NUM_WORKER,
        'shuffle': cfg.DATALOADER.SHUFFLE,
        'drop_last': True,
        'pin_memory': True,
    }
    labeled_dataloader = torch_data.DataLoader(labeled_dataset, **dataloader_kwargs)
    unlabeled_dataloader = torch_data.DataLoader(unlabeled_dataset, **dataloader_kwargs)

    # unpacking cfg
    epochs = cfg.TRAINER.EPOCHS

    # tracking variables
    global_step = epoch_float = 0

    # early stopping
    best_f1_val, trigger_times = 0, 0
    stop_training = False

    steps_per_epoch = len(unlabeled_dataloader)

    for epoch in range(1, epochs + 1):
        print(f'Starting epoch {epoch}/{epochs}.')

        start = timeit.default_timer()
        change_loss_set, sup_loss_set, cons_loss_set, loss_set = [], [], [], []
        dataloader = iter(zip(helpers.cycle(labeled_dataloader), unlabeled_dataloader))

        for i, (labeled_batch, unlabeled_batch) in enumerate(dataloader):

            net.train()
            optimizer.zero_grad()

            # supervised loss
            x_t1_l = labeled_batch['x_t1'].to(device)
            x_t2_l = labeled_batch['x_t2'].to(device)
            y_change = labeled_batch['y_change'].to(device)
            # print(f'{i}/{steps_per_epoch} {torch.sum(y_change).item()}')

            logits_change_l, _ = net(x_t1_l, x_t2_l)

            change_loss = sup_criterion(logits_change_l, y_change)
            sup_loss = change_loss
            change_loss_set.append(change_loss.item())
            sup_loss_set.append(sup_loss.item())

            # unsupervised loss
            x_t1_ul = unlabeled_batch['x_t1'].to(device)
            x_t2_ul = unlabeled_batch['x_t2'].to(device)

            logits_change_ul, logits_change_noisy_ul = net(x_t1_ul, x_t2_ul)
            y_hat_change_noisy_ul = torch.sigmoid(logits_change_noisy_ul)
            if cfg.CONSISTENCY_TRAINER.LOSS_TYPE == 'L2':
                y_hat_change_ul = torch.sigmoid(logits_change_ul)
                cons_loss = cons_criterion(y_hat_change_ul, y_hat_change_noisy_ul)
            else:
                cons_loss = cons_criterion(logits_change_ul, y_hat_change_noisy_ul)
            cons_loss = cons_loss * cfg.CONSISTENCY_TRAINER.LOSS_FACTOR
            cons_loss_set.append(cons_loss.item())

            loss = sup_loss + cons_loss
            loss_set.append(loss.item())

            loss.backward()
            optimizer.step()

            global_step += 1
            epoch_float = global_step / steps_per_epoch

            if global_step % cfg.LOGGING.FREQUENCY == 0:
                print(f'Logging step {global_step} (epoch {epoch_float:.2f}).')
                time = timeit.default_timer() - start
                wandb.log({
                    'change_loss': np.mean(change_loss_set),
                    'sup_loss': np.mean(sup_loss_set),
                    'cons_loss': np.mean(cons_loss_set),
                    'loss': np.mean(loss_set),
                    'labeled_percentage': 50,
                    'time': time,
                    'step': global_step,
                    'epoch': epoch_float,
                })
                start = timeit.default_timer()
                change_loss_set, sup_loss_set, cons_loss_set, loss_set = [], [], [], []
            # end of batch

        assert (epoch == epoch_float)
        print(f'epoch float {epoch_float} (step {global_step}) - epoch {epoch}')
        # evaluation at the end of an epoch
        # _ = evaluation.model_evaluation(net, cfg, 'train', epoch_float, global_step)
        f1_val = evaluation.model_evaluation(net, cfg, 'val', epoch_float, global_step)

        if f1_val <= best_f1_val:
            trigger_times += 1
            if trigger_times > cfg.TRAINER.PATIENCE:
                stop_training = True
        else:
            best_f1_val = f1_val
            wandb.log({
                'best val change F1': best_f1_val,
                'step': global_step,
                'epoch': epoch_float,
            })
            print(f'saving network (F1 {f1_val:.3f})', flush=True)
            networks.save_checkpoint(net, optimizer, epoch, cfg)
            trigger_times = 0

        if stop_training:
            break  # end of training by early stopping

    net, *_ = networks.load_checkpoint(cfg, device)
    _ = evaluation.model_evaluation(net, cfg, 'test', epoch_float, global_step)


if __name__ == '__main__':
    args = parsers.training_argument_parser().parse_known_args()[0]
    cfg = experiment_manager.setup_cfg(args)

    # make training deterministic
    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    random.seed(cfg.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

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
