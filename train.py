import os
import model
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Loss
import logging
from dataset import getResiscData
from common import Cfg128

global ITER
ITER = 0

cwd = os.getcwd()
expr_out = os.path.join(cwd, 'experiments', 'on_resisc45')
logging.basicConfig(filename=os.path.join(expr_out, 'eval_log'),
                    level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('train_logger')
train_dataset, val_dataset = getResiscData()
logger.info('Load trainset: ' + str(len(train_dataset)))
logger.info('Load validset: ' + str(len(val_dataset)))
logger.info('Start training.')

cfg = Cfg128()
homog_net = model.HomographyNet(cfg)
train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=True)
optimizer = cfg.get_optimizer(homog_net.parameters())
scheduler = cfg.get_lr_scheduler(optimizer)
loss = cfg.get_loss()

trainer = create_supervised_trainer(homog_net, optimizer, loss, device=cfg.device)
evaluator = create_supervised_evaluator(homog_net, metrics={'esti_error': Loss(loss)})
checkpointer = ModelCheckpoint(expr_out, 'homog_net', cfg.save_period, n_saved=10, require_empty=False)

trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer,
                          {'model': homog_net, 'optimizer': optimizer})


@trainer.on(Events.ITERATION_COMPLETED)
def log_training_loss(trainer):
    global ITER
    ITER += 1
    if ITER % cfg.log_period == 0:
        logger.info("Epoch[{}] Loss: {:.2f}".format(trainer.state.epoch, trainer.state.output))
    if len(train_loader) == ITER:
        ITER = 0


@trainer.on(Events.EPOCH_STARTED)
def adjust_lr(trainer):
    scheduler.step()


@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(trainer):
    evaluator.run(train_loader)
    metrics = evaluator.state.metrics
    logger.info("Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
                .format(trainer.state.epoch, metrics['accuracy'], metrics['nll']))


@trainer.on(Events.EPOCH_COMPLETED)
def log_validation_results(trainer):
    evaluator.run(val_loader)
    metrics = evaluator.state.metrics
    logger.info("Validation Results - Epoch: {} "
                .format(trainer.state.epoch, metrics['esti_error']))


trainer.run(train_loader, max_epochs=cfg.max_epoch)
