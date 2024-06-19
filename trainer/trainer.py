import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker
from model.model import ViT, CLIP
from model.model import Classifier


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)

        # pre-trained model
        self.ViT = ViT()
        self.CLIP = CLIP()
        
        self.Classifier = Classifier()

        self.config = config
        self.device = device
        self.data_loader = data_loader
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
        self.log_step = int(np.sqrt(data_loader.batch_size))

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
        for batch_idx, (data, target) in enumerate(self.data_loader):
            # data, target = data.to(self.device), target.to(self.device)
            image, representation, attn = data
            representation, attn, target = representation.to(self.device), attn.to(self.device), target.to(self.device)
            # print(representation.shape, attn.shape, target.shape)

            self.optimizer.zero_grad()
            W, delta, assign, M = self.model(representation) # assign(B, token): 每个token属于的concept
            assign = assign.detach().to(self.device)
            loss = self.criterion(W.detach().to(self.device), delta, M.detach().to(self.device))
            loss.backward()
            self.optimizer.step()

            # Split background and foreground
            split_res = self.Classifier.SplitBackgroud(assign, attn) # (B, concepts) == (B, 5)
            output = np.zeros_like(image.shape)
            bsz = split_res.shape[0]
            for i in range(bsz):
                foreground = [idx for idx, val in enumerate(split_res[i]) if val == 1]
                max_concept = max(foreground)
                tokens = [idx for idx, val in enumerate(assign[i]) if val in foreground]
                o = torch.zeros(16, 16)
                for token in tokens:
                    o[tokens] = assign[i][token]
                # linear interpolation
                o = torch.nn.functional.interpolate(o, size=(224, 224), mode='bilinear')
                mask = []
                for i in range(max_concept):
                    i_mask = torch.where(o > i and o <= i + 1, torch.tensor(1), torch.tensor(0))
                    mask.append(i_mask)
                output[i] = o

            # CLIP for zero_shot


            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))
                self.writer.add_image('input', make_grid(image.cpu(), nrow=8, normalize=True))
                self.writer.add_image('output', make_grid(output.cpu(), nrow=8, normalize=True))
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8))
                # semantic_reult = make_mask(data, )
                # self.writer.add_image('output', make_grid(output.cpu(), nrow=8, normalize=True))


            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

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
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

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
