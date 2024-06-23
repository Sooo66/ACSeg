import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker, Visualize, apply_mask_with_transparency
from model.model import Classifier
from tqdm import tqdm

class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        
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

        pbar = tqdm(enumerate(self.data_loader), total=self.len_epoch, desc=f'(Train)Epoch {epoch}', leave=True, ncols=100)
        key, value = [], []

        # for batch_idx, (data, target) in enumerate(self.data_loader):
        for batch_idx, (data, target) in pbar:
            image, representation, attn = data
            # image -> float32
            # image = image.to(torch.float32)
            representation, attn = representation.to(self.device), attn.to(self.device)
            mt, vt = target
            mt, vt = mt.to(self.device), vt.to(self.device)
            mt[mt == 255] = 0
            # target: PIL

            self.optimizer.zero_grad()
            concepts, W, delta, S, M = self.model(representation) # S: (B, tokens, concepts)
            loss = self.criterion(W.detach().to(self.device), delta, M.detach().to(self.device))
            loss.backward()
            self.optimizer.step()

            assign = torch.argmax(S, dim=2) # (B, tokens)
            foreground = self.Classifier.SplitBackgroud(assign, attn) # (B, concepts) == (B, 5)
            mask = self.Classifier.GetMask(S, foreground) # (B, 224, 224)
            k, v = self.Classifier.CreateKNN(mask, mt, concepts)
            key += k
            value += v

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.detach().item())

            # self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
            #         epoch,
            #         self._progress(batch_idx),
            #         loss.detach().item()))
            # for met in self.metric_ftns:
            #     self.train_metrics.update(met.__name__, met(output, target))
            pbar.update(1)

            if batch_idx % self.log_step == 0:
                # Split background and foreground
                assign = torch.argmax(S, dim=2) # (B, tokens)
                foreground = self.Classifier.SplitBackgroud(assign, attn) # (B, concepts) == (B, 5)
                mask = self.Classifier.GetMask(S, foreground) # (B, 224, 224)
                bbxs = self.Classifier.GetBox(mask, image) # (B, [])

                # for met in self.metric_ftns:
                    # self.train_metrics.update(met.__name__, met(output, mt))

                bbxs_img0 = [] # 每个bsz只扣一张看效果
                bbxs_img1 = []
                for i in range(len(bbxs)): # bsz
                    bbx = bbxs[i]
                    if len(bbx) > 1:
                        bbxs_img0.append(torch.from_numpy(bbx[0][0]).float())
                        bbxs_img1.append(torch.from_numpy(bbx[1][0]).float())
                    else :
                        bbxs_img0.append(torch.zeros(224, 224, 3))
                        bbxs_img1.append(torch.zeros(224, 224, 3))
                bbxs_img0 = torch.stack(bbxs_img0)
                bbxs_img1 = torch.stack(bbxs_img1)

                bsz = mask.shape[0]
                cpts = []
                for i in range(bsz):
                    mk = mask[i].clone()
                    im = apply_mask_with_transparency(image[i], mk)
                    cpts.append(im)
                cpts = torch.stack(cpts, dim=0)

                self.writer.add_image('input', make_grid(image.detach().cpu(), nrow=8, normalize=True))
                self.writer.add_image('concepts', make_grid(cpts.detach().cpu(), nrow=8))

                self.writer.add_image('bbx0', make_grid(torch.permute(bbxs_img0, (0, 3, 1, 2)).detach().cpu(), nrow=8))
                self.writer.add_image('bbx1', make_grid(torch.permute(bbxs_img1, (0, 3, 1, 2)).detach().cpu(), nrow=8))
                self.writer.add_image('target', make_grid(vt.detach().cpu(), nrow=8)) # target(B, 224, 224)


            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            key = torch.stack(key, dim=0)
            value = torch.stack(value, dim=0)
            val_log = self._valid_epoch(epoch, key, value)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch, key, value):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        vpbar = tqdm(enumerate(self.valid_data_loader), total=len(self.valid_data_loader), desc=f'(Valid)Epoch {epoch}', leave=True, ncols=100)
        with torch.no_grad():
            for batch_idx, (data, target) in vpbar:
                # data, target = data.to(self.device), target.to(self.device)

                image, representation, attn = data
                # image -> float32
                # image = image.to(torch.float32)
                representation, attn = representation.to(self.device), attn.to(self.device)
                mt, vt = target
                mt, vt = mt.to(self.device), vt.to(self.device)
                mt[mt == 255] = 0

                # --------
                self.optimizer.zero_grad()
                concepts, W, delta, S, M = self.model(representation) # S: (B, tokens, concepts)
                loss = self.criterion(W.detach().to(self.device), delta, M.detach().to(self.device))

                assign = torch.argmax(S, dim=-1) # (B, tokens)
                foreground = self.Classifier.SplitBackgroud(assign, attn) # (B, concepts) == (B, 5)
                mask = self.Classifier.GetMask(S, foreground) # (B, 224, 224)
                output = self.Classifier.KnnRetrivel(concepts, key, value, mask)
                output = torch.stack(output, dim=0)
                # --------

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.detach().item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, mt))
                self.writer.add_image('valid_input', make_grid(image.detach().cpu(), nrow=8, normalize=True))
                o = torch.from_numpy(Visualize(output)).float()
                self.writer.add_image('valid_output', make_grid(torch.permute(o, (0, 3, 1, 2)).detach().cpu(), nrow=8))
                vpbar.update(1)

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
