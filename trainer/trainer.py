from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.data_loader = data_loader
        self.seq2seq_criterion = nn.NLLLoss()
        
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(self.len_epoch / 4) # int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        trange = tqdm(enumerate(self.data_loader), total=self.len_epoch, desc="training")
        for batch_idx, batch in trange:
            # bos_sentence = batch["bos_sentence"]
            # eos_sentence = batch["eos_sentence"]
            source = batch['source']
            target = batch['target']
            target_bos = batch['target_bos']
            target_eos = batch['target_eos']
   
            source = source.to(self.device)
            target = target.to(self.device)
            target_bos = target_bos.to(self.device)
            target_eos = target_eos.to(self.device)


            self.optimizer.zero_grad()
            # score, dec_output = self.model(bos_sentence, eos_sentence)
            output = self.model(source, target_bos) #[batch, seq_len, vocab_size]
            # print('output.size: ', output.size())

            loss = self.criterion(output, target_eos).float()
            # loss = self.criterion(output, target).float()
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())

            predict = output.max(2)[1]
            #print(predict)
            #predict = predict.type(torch.LongTensor).to(self.device)

            for met in self.metric_ftns:
                if met.__name__ == 'bleu_score': self.train_metrics.update(met.__name__, met(predict, target_eos, self.data_loader.embedding))
                else: self.train_metrics.update(met.__name__, met(predict, target_eos))

            
            '''
            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))
            '''
            if batch_idx == self.len_epoch:
                break
            trange.set_postfix(loss=loss.item())
        log = self.train_metrics.result()

        print('> : ' + self.data_loader.embedding.indice_to_sentence(source[0].tolist()))
        print('= : ' + self.data_loader.embedding.indice_to_sentence(target[0].tolist()))
        print('< : ' + self.data_loader.embedding.indice_to_sentence(predict[0].tolist()))

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.valid_data_loader):
                source = batch['source']
                target = batch['target']
                target_bos = batch['target_bos']
                target_eos = batch['target_eos']
                source = source.to(self.device)
                target = target.to(self.device)
                target_bos = target_bos.to(self.device)
                target_eos = target_eos.to(self.device)

                output = self.model(source, target_bos)
                loss = self.criterion(output, target_eos).float()
                #loss = self.criterion(output, target).float()

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())

                predict = output.max(2)[1]

                for met in self.metric_ftns:
                    if met.__name__ == 'bleu_score': self.valid_metrics.update(met.__name__, met(predict, target_eos, self.data_loader.embedding))
                    else: self.valid_metrics.update(met.__name__, met(predict, target_eos))
                #self.writer.add_image('input', make_grid(predict.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)


