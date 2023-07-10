import sys
import os
import timeit

import torch
from torch import optim
from torch.utils import data as torch_data

import wandb
import numpy as np

from utils import networks, datasets, loss_functions, evaluation, experiment_manager, parsers, helpers


def run_training(cfg):
    net = networks.create_network(cfg)
    net.to(device)
    optimizer = optim.AdamW(net.parameters(), lr=cfg.TRAINER.LR, weight_decay=0.01)

    sup_criterion = loss_functions.get_criterion(cfg.MODEL.LOSS_TYPE)
    cons_criterion = loss_functions.get_criterion(cfg.CONSISTENCY_TRAINER.LOSS_TYPE)

    dataloader_kwargs = {
        'batch_size': cfg.TRAINER.BATCH_SIZE,
        'num_workers': 0 if cfg.DEBUG else cfg.DATALOADER.NUM_WORKER,
        'shuffle': cfg.DATALOADER.SHUFFLE,
        'drop_last': True,
        'pin_memory': True,
    }

    # unpacking cfg
    epochs = cfg.TRAINER.EPOCHS
    warmup_epochs = cfg.TRAINER.WARMUP_EPOCHS

    # tracking variables
    global_step = epoch_float = 0

    # early stopping
    best_f1_val, trigger_times = 0, 0
    stop_training = False

    # reset the generators
    warmup_dataset = datasets.MultimodalCDDataset(cfg=cfg, run_type='train', disable_unlabeled=True)
    print(warmup_dataset)
    warmup_dataloader = torch_data.DataLoader(warmup_dataset, **dataloader_kwargs)
    steps_per_warmup_epoch = len(warmup_dataloader)

    for epoch in range(1, warmup_epochs + 1):
        print(f'Starting epoch {epoch}/{epochs} (warmup epoch {epoch}/{warmup_epochs}).')

        start = timeit.default_timer()
        change_loss_set, sem_loss_set, sup_loss_set, loss_set = [], [], [], []

        for i, batch in enumerate(warmup_dataloader):

            net.train()
            optimizer.zero_grad()

            x_t1 = batch['x_t1'].to(device)
            x_t2 = batch['x_t2'].to(device)

            logits_change, logits_sem_t1, logits_sem_t2 = net(x_t1, x_t2)
            logits_change_sem = net.module.outc_change_sem(torch.cat((logits_sem_t1, logits_sem_t2), dim=1))

            # change detection
            y_change = batch['y_change'].to(device)
            change_loss = sup_criterion(logits_change, y_change)
            change_sem_loss = sup_criterion(logits_change_sem, y_change)

            # semantic segmentation
            y_sem_t1, y_sem_t2 = batch['y_sem_t1'].to(device), batch['y_sem_t2'].to(device)

            sem_t1_loss = sup_criterion(logits_sem_t1, y_sem_t1)
            sem_t2_loss = sup_criterion(logits_sem_t2, y_sem_t2)

            change_loss = change_loss + change_sem_loss
            sem_loss = (sem_t1_loss + sem_t2_loss) / 2
            sup_loss = change_loss + sem_loss

            change_loss_set.append(change_loss.item())
            sem_loss_set.append(sem_loss.item())
            sup_loss_set.append(sup_loss.item())

            loss = sup_loss
            loss_set.append(loss.item())

            loss.backward()
            optimizer.step()

            global_step += 1
            epoch_float = global_step / steps_per_warmup_epoch

            if global_step % cfg.LOGGING.FREQUENCY == 0:
                print(f'Logging step {global_step} (epoch {epoch_float:.2f}).')
                time = timeit.default_timer() - start
                wandb.log({
                    'change_loss': np.mean(change_loss_set) if len(change_loss_set) > 0 else 0,
                    'sem_loss': np.mean(sem_loss_set) if len(sem_loss_set) > 0 else 0,
                    'sup_loss': np.mean(sup_loss_set) if len(sup_loss_set) > 0 else 0,
                    'cons_loss': 0,
                    'loss': np.mean(loss_set),
                    'labeled_percentage': 100,
                    'time': time,
                    'step': global_step,
                    'epoch': epoch_float,
                })
                start = timeit.default_timer()
                change_loss_set, sem_loss_set, sup_loss_set, loss_set = [], [], [], []
            # end of batch

        assert (epoch == epoch_float)
        print(f'epoch float {epoch_float} (step {global_step}) - epoch {epoch}')
        # evaluation at the end of an epoch
        # _ = evaluation.model_evaluation_dt(net, cfg, 'train', epoch_float, global_step)
        f1_val = evaluation.model_evaluation_dt(net, cfg, 'val', epoch_float, global_step)

        if f1_val <= best_f1_val:
            trigger_times += 1
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

    # reset the generators
    labeled_dataset = datasets.MultimodalCDDataset(cfg=cfg, run_type='train', disable_unlabeled=True)
    unlabeled_dataset = datasets.MultimodalCDDataset(cfg=cfg, run_type='train', only_unlabeled=True)
    print(labeled_dataset, unlabeled_dataset)
    dataloader_kwargs['batch_size'] = cfg.TRAINER.BATCH_SIZE // 2
    labeled_dataloader = torch_data.DataLoader(labeled_dataset, **dataloader_kwargs)
    unlabeled_dataloader = torch_data.DataLoader(unlabeled_dataset, **dataloader_kwargs)

    steps_per_epoch = len(unlabeled_dataloader)

    for epoch in range(warmup_epochs + 1, epochs + 1):
        print(f'Starting epoch {epoch}/{epochs}.')

        start = timeit.default_timer()
        change_loss_set, sem_loss_set, sup_loss_set, cons_loss_set, loss_set = [], [], [], [], []
        dataloader = iter(zip(helpers.cycle(labeled_dataloader), unlabeled_dataloader))

        for i, (labeled_batch, unlabeled_batch) in enumerate(dataloader):

            net.train()
            optimizer.zero_grad()

            # supervised loss
            x_t1_l, x_t2_l = labeled_batch['x_t1'].to(device), labeled_batch['x_t2'].to(device)
            y_change = labeled_batch['y_change'].to(device)
            y_sem_t1, y_sem_t2 = labeled_batch['y_sem_t1'].to(device), labeled_batch['y_sem_t2'].to(device)

            logits_l_change, logits_l_sem_t1, logits_l_sem_t2 = net(x_t1_l, x_t2_l)
            logits_l_change_sem = net.module.outc_change_sem(torch.cat((logits_l_sem_t1, logits_l_sem_t2), dim=1))

            # change detection
            change_loss = sup_criterion(logits_l_change, y_change)
            change_sem_loss = sup_criterion(logits_l_change_sem, y_change)

            # semantic segmentation
            sem_t1_loss = sup_criterion(logits_l_sem_t1, y_sem_t1)
            sem_t2_loss = sup_criterion(logits_l_sem_t2, y_sem_t2)

            change_loss = change_loss + change_sem_loss
            sem_loss = (sem_t1_loss + sem_t2_loss) / 2
            sup_loss = change_loss + sem_loss

            change_loss_set.append(change_loss.item())
            sem_loss_set.append(sem_loss.item())
            sup_loss_set.append(sup_loss.item())

            # unsupervised loss
            x_t1_ul, x_t2_ul = unlabeled_batch['x_t1'].to(device), unlabeled_batch['x_t2'].to(device)
            logits_ul_change, logits_ul_sem_t1, logits_ul_sem_t2 = net(x_t1_ul, x_t2_ul)
            logits_ul_change_sem = net.module.outc_change_sem(torch.cat((logits_ul_sem_t1, logits_ul_sem_t2), dim=1))
            y_hat_ul_change_sem = torch.sigmoid(logits_ul_change_sem)

            if cfg.CONSISTENCY_TRAINER.LOSS_TYPE == 'L2':
                y_hat_ul_change = torch.sigmoid(logits_ul_change)
                cons_loss = cons_criterion(y_hat_ul_change, y_hat_ul_change_sem)
            else:
                cons_loss = cons_criterion(logits_ul_change, y_hat_ul_change_sem)
            cons_loss = cons_loss * cfg.CONSISTENCY_TRAINER.LOSS_FACTOR
            cons_loss_set.append(cons_loss.item())

            loss = sup_loss + cons_loss
            loss_set.append(loss.item())

            loss.backward()
            optimizer.step()

            global_step += 1
            epoch_float = warmup_epochs + (global_step - warmup_epochs * steps_per_warmup_epoch) / steps_per_epoch

            if global_step % cfg.LOGGING.FREQUENCY == 0:
                print(f'Logging step {global_step} (epoch {epoch_float:.2f}).')
                time = timeit.default_timer() - start
                wandb.log({
                    'change_loss': np.mean(change_loss_set) if len(change_loss_set) > 0 else 0,
                    'sem_loss': np.mean(sem_loss_set) if len(sem_loss_set) > 0 else 0,
                    'sup_loss': np.mean(sup_loss_set) if len(sup_loss_set) > 0 else 0,
                    'cons_loss': np.mean(cons_loss_set) if len(cons_loss_set) > 0 else 0,
                    'loss': np.mean(loss_set),
                    'labeled_percentage': 50,
                    'time': time,
                    'step': global_step,
                    'epoch': epoch_float,
                })
                start = timeit.default_timer()
                change_loss_set, sem_loss_set, sup_loss_set, cons_loss_set, loss_set = [], [], [], [], []
            # end of batch

        assert (epoch == epoch_float)
        print(f'epoch float {epoch_float} (step {global_step}) - epoch {epoch}')
        # evaluation at the end of an epoch
        # _ = evaluation.model_evaluation_dt(net, cfg, 'train', epoch_float, global_step)
        f1_val = evaluation.model_evaluation_dt(net, cfg, 'val', epoch_float, global_step)

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
    _ = evaluation.model_evaluation_dt(net, cfg, 'test', epoch_float, global_step)


if __name__ == '__main__':
    args = parsers.training_argument_parser().parse_known_args()[0]
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