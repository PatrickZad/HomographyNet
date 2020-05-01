import os
import model
from torch.utils.data import DataLoader

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Loss
import logging
from dataset import getResiscData
from common import Cfg128
from metric import MeanAveragePosError

global ITER


def train_from_scratch(cfg):
    global ITER
    ITER = 0

    cwd = os.getcwd()
    expr_out = os.path.join(cwd, 'experiments', 'on_resisc45')
    logging.basicConfig(filename=os.path.join(expr_out, 'train_log'),
                        level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('train_logger')

    train_dataset, val_dataset = getResiscData(device=cfg.device)
    loc_error = MeanAveragePosError((cfg.width_in, cfg.height_in), cfg.device)
    homog_net = model.HomographyNet(cfg)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=True)

    logger.info('Load trainset: ' + str(len(train_dataset)))
    logger.info('Load validset: ' + str(len(val_dataset)))
    logger.info('Start training.')

    optimizer = cfg.get_optimizer(homog_net.parameters())
    scheduler = cfg.get_lr_scheduler(optimizer)
    loss = cfg.get_loss()

    trainer = create_supervised_trainer(homog_net, optimizer, loss, device=cfg.device)
    evaluator = create_supervised_evaluator(homog_net, metrics={'esti_error': Loss(loss), 'loc_error': loc_error},
                                            device=cfg.device)
    checkpointer = ModelCheckpoint(expr_out, 'homog_net', n_saved=10, require_empty=False)

    trainer.add_event_handler(Events.EPOCH_COMPLETED(every=cfg.save_period), checkpointer,
                              {'model': homog_net, 'optimizer': optimizer})

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(trainer):
        global ITER
        ITER += 1
        if ITER % cfg.log_period == 0:
            logger.info("Epoch[{}] Batch[{}] Loss: {:.2f}".format(trainer.state.epoch, ITER, trainer.state.output))
            print("Epoch[{}] Batch[{}] Loss: {:.2f}".format(trainer.state.epoch, ITER, trainer.state.output))
        if len(train_loader) == ITER:
            ITER = 0

    @trainer.on(Events.EPOCH_COMPLETED)
    def adjust_lr(trainer):
        scheduler.step()

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        logger.info("Validation Results - Epoch: {} Val_loss: {},Loc_error: {}"
                    .format(trainer.state.epoch, metrics['esti_error'], metrics['loc_error']))
        print("Validation Results - Epoch: {} Val_loss: {} Loc_error: {}"
              .format(trainer.state.epoch, metrics['esti_error'], metrics['loc_error']))

    trainer.run(train_loader, max_epochs=cfg.max_epoch)


if __name__ == '__main__':
    cfg = Cfg128()
    train_from_scratch(cfg)
