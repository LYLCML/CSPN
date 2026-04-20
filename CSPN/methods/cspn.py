import methods.wideresnet as wideresnet
from methods.augtools import HighlyCustomizableAugment, RandAugmentMC
import methods.util as util
import methods.Backbone_method as Backbone_method
import tqdm
from sklearn import metrics
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data as data
import random
from methods.util import AverageMeter
import time
from torchvision.transforms import transforms
from methods.resnet import ResNet
from torchvision import models as torchvision_models
import numpy as np
import metrics


@util.regmethod('cspn')
class CSPNMethod:

    def __init__(self, config, clssnum, train_set) -> None:
        self.config = config
        self.epoch = 0
        self.lr = config['learn_rate']
        self.batch_size = config['batch_size']
        self.clsnum = clssnum
        self.crt = CSPNCriterion()
        self.model = CSPNModel(self.clsnum, config, self.crt).cuda()
        self.modelopt = torch.optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=5e-4)
        self.wrap_ds = WrapDataset(train_set, self.config, inchan_num=3, )
        self.wrap_loader = data.DataLoader(self.wrap_ds,
                                           batch_size=self.config['batch_size'], shuffle=True, pin_memory=True,
                                           num_workers=6)
        self.lr_schedule = util.get_scheduler(self.config, self.wrap_loader)

    def train_epoch(self, flag):
        data_time = AverageMeter()
        batch_time = AverageMeter()
        train_acc = AverageMeter()
        running_loss = AverageMeter()
        self.model.train()
        endtime = time.time()
        for i, data in enumerate(tqdm.tqdm(self.wrap_loader)):
            data_time.update(time.time() - endtime)
            self.lr = self.lr_schedule.get_lr(self.epoch, i, self.lr)
            util.set_lr([self.modelopt], self.lr)
            sx, lb = data[0].cuda(), data[1].cuda()
            loss, lossp, pred, scores, logic_distance = self.model(sx, lb, reqpredauc=True)
            if flag == 1:
                loss = loss + 0.1* lossp
            if flag == 0:
                loss = loss + 0.1* lossp
            self.modelopt.zero_grad()
            loss.backward()
            self.modelopt.step()
            nplb = data[1].numpy()
            train_acc.update((pred == nplb).sum() / pred.shape[0], pred.shape[0])
            running_loss.update(loss.item())
            batch_time.update(time.time() - endtime)
            endtime = time.time()
        print(self.model.backbone_cs.cspn_cls.R_c)
        self.epoch += 1
        training_res = \
            {"Loss": running_loss.avg,
             "TrainAcc": train_acc.avg,
             "Learn Rate": self.lr,
             "DataTime": data_time.avg,
             "BatchTime": batch_time.avg}
        return training_res


    def scoring(self, loader, prepare=False):
        gts = []
        logic = []
        lts_X = []
        prediction = []
        with (torch.no_grad()):
            for d in tqdm.tqdm(loader):
                x1 = d[0].cuda(non_blocking=True)
                gt = d[1].numpy()
                pred, scr, dev = self.model(x1, reqpredauc=True, prepareTest=prepare)
                xcloreF,lt = self.model(x1, reqpredauc=False, prepareTest=prepare)
                prediction.append(pred)
                gts.append(gt)
                logic.append(xcloreF)
                lts_X.append(lt)
        lts_X = [x.cpu().numpy() if isinstance(x, torch.Tensor) else x for x in lts_X]
        lts_X = np.concatenate(lts_X)
        prediction = np.concatenate(prediction)
        gts = np.concatenate(gts)
        logic = [x.cpu().numpy() if isinstance(x, torch.Tensor) else x for x in logic]
        logic = np.concatenate(logic)
        return prediction, gts, logic,lts_X

    def knownpred_unknwonscore_test(self, test_loader):
        self.model.eval()
        pred_test, gts_test, scores_test_kn, laten_test = self.scoring(test_loader)
        activate_dastance = scores_test_kn[np.arange(scores_test_kn.shape[0]), pred_test]
        return activate_dastance, -9999999, pred_test

    def save_model(self, path):
        save_dict = {
            'model': self.model.state_dict(),
            'config': self.config,
            'optimzer' : self.modelopt.state_dict(),
            'epoch' : self.epoch,
        }
        torch.save(save_dict,path)

    def load_model(self,path):
        save_dict = torch.load(path)
        self.model.load_state_dict(save_dict['model'])
        if 'optimzer' in save_dict and self.modelopt is not None:
            self.modelopt.load_state_dict(save_dict['optimzer'])
        self.epoch = save_dict['epoch']

def sim_conv_layer(input_channel, output_channel, kernel_size=1, padding=0, use_activation=True):
    if use_activation:
        res = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size, padding=padding, bias=False),
            nn.Tanh())
    else:
        res = nn.Conv2d(input_channel, output_channel, kernel_size, padding=padding, bias=False)
    return res


class Encoder(nn.Module):
    def __init__(self, inchannel, latent_chan):
        super().__init__()
        layer_block = sim_conv_layer
        self.latent_size = latent_chan
        hidd = 64
        self.extra_layer = layer_block(inchannel, hidd)
        self.latent_conv = layer_block(hidd, latent_chan)

    def forward(self, x):
        output = x
        output = self.extra_layer(output)
        output = self.latent_conv(output)
        latent = output.view(output.size(0), -1)
        return latent

class CSPNClassifier(nn.Module):

    def __init__(self, inchannels, num_class, config):
        super().__init__()
        en_latent = config['en_latent']
        en_H_W = config['en_H_W']
        self.class_enc = []
        for i in range(num_class):
            en = Encoder(inchannels, en_latent)
            self.class_enc.append(en)
        self.class_enc = nn.ModuleList(self.class_enc)
        self.prototypes = nn.Parameter(torch.randn(num_class, en_latent*en_H_W*en_H_W))
        self.register_buffer('R_c', torch.tensor(config['R_c']))

    def prototypes_error(self, lt, prototype):
        prototype_expanded = prototype.unsqueeze(0)
        return torch.norm(lt - prototype_expanded, p=2, dim=1, keepdim=True)
    clip_len = 100

    def forward(self, x, ycls):
        cls_ers = []
        lts = []
        aa=[]
        for c in range(len(self.class_enc)):
            lt = self.class_enc[c](x)
            cls_er = self.prototypes_error(lt, self.prototypes[c])
            aa.append(cls_er)
            if self.training:
                same_class_mask = (ycls == c)
                cls_er = torch.where(
                    same_class_mask.unsqueeze(1),
                    cls_er,
                    torch.clamp(cls_er, max=100)
                )
            cls_ers.append(-cls_er)
            lts.append(lt)
        logits = torch.cat(cls_ers, dim=1)
        lts = torch.stack(lts, dim=1)
        diff =  self.prototypes - 0
        dist = torch.norm(diff, p=2, dim=1)
        lossp = ((dist - self.R_c) ** 2).mean()
        return logits, lossp,lts


class BackboneAndClassifier(nn.Module):

    def __init__(self, num_classes, config):
        super().__init__()
        self.backbone = Backbone_method.Backbone(config, 3)
        cspn_config = config['cspn_model']
        cspn_config["R_c"] = config['R_c']
        self.cspn_cls = CSPNClassifier(self.backbone.output_dim, num_classes, cspn_config)

    def forward(self, x, ycls, feature_only=False):
        x = self.backbone(x)
        if feature_only:
            return x
        logic, lossp, lts = self.cspn_cls(x, ycls)
        return x, logic, lossp,lts


class CSPNModel(nn.Module):

    def __init__(self, num_classes, config, crt):
        super().__init__()
        self.crt = crt
        self.backbone_cs = BackboneAndClassifier(num_classes, config)
        self.config = config
        self.num_classes = num_classes

    def forward(self, x, ycls=None, reqpredauc=False, prepareTest=False,
                reqfeature=False):

        x, xcls_raw, lossp,lts = self.backbone_cs(x, ycls, feature_only=reqfeature)
        if reqfeature:
            return x

        def pred_score(xcls):
            score_reduce = lambda x: x.reshape([x.shape[0], -1]).mean(axis=1)
            x_detach = x.detach()
            probs = self.crt(xcls, prob=True).cpu().numpy()
            pred = probs.argmax(axis=1)
            rep_scores = torch.abs(x_detach).mean(dim=1).cpu().numpy()
            return pred, score_reduce(rep_scores)

        if self.training:
            loss = self.crt(xcls_raw, ycls)
            if reqpredauc:
                pred, score = pred_score(xcls_raw.detach())
                return loss, lossp, pred, score, xcls_raw
        else:
            xcls = xcls_raw
            if reqpredauc:
                pred, score = pred_score(xcls)
                deviations = None
                return pred, score, deviations
        return xcls,lts


class CSPNCriterion(nn.Module):
    def get_onehot_label(self, y, clsnum):
        y = torch.reshape(y, [-1, 1]).long()
        return torch.zeros(y.shape[0], clsnum).cuda().scatter_(1, y, 1)

    def __init__(self):
        super().__init__()

    def forward(self, x, y=None, prob=False, pred=False):
        g = torch.softmax(x, dim=1)
        if prob: return g
        if pred: return torch.argmax(g, dim=1)
        loss = -torch.sum(self.get_onehot_label(y, g.shape[1]) * torch.log(g), dim=1).mean()
        return loss


def manual_contrast(x):
    s = random.uniform(0.1, 2)
    return x * s


class WrapDataset(data.Dataset):

    def __init__(self, labeled_ds, config, inchan_num=3) -> None:
        super().__init__()
        self.labeled_ds = labeled_ds
        __mean = [0.5, 0.5, 0.5][:inchan_num]
        __std = [0.25, 0.25, 0.25][:inchan_num]
        trans = [transforms.RandomHorizontalFlip()]
        if config['cust_aug_crop_withresize']:
            trans.append(transforms.RandomResizedCrop(size=util.img_size, scale=(0.25, 1)))
        elif util.img_size > 200:
            trans += [transforms.Resize(256), transforms.RandomResizedCrop(util.img_size)]
        else:
            trans.append(transforms.RandomCrop(size=util.img_size,
                                               padding=int(util.img_size * 0.125),
                                               padding_mode='reflect'))
        if config['strong_option'] == 'RA':
            trans.append(RandAugmentMC(n=2, m=10))
        elif config['strong_option'] == 'CUST':
            trans.append(HighlyCustomizableAugment(2, 10, -1, labeled_ds, config))
        elif config['strong_option'] == 'NONE':
            pass
        else:
            raise NotImplementedError()
        trans += [transforms.ToTensor(),
                  transforms.Normalize(mean=__mean, std=__std)]
        if config['manual_contrast']:
            trans.append(manual_contrast)
        strong = transforms.Compose(trans)
        if util.img_size > 200:
            self.simple = [transforms.RandomResizedCrop(util.img_size)]
        else:
            self.simple = [transforms.RandomCrop(size=util.img_size,
                                                 padding=int(util.img_size * 0.125),
                                                 padding_mode='reflect')]
        self.simple = transforms.Compose(([transforms.RandomHorizontalFlip()]) + self.simple + [
            transforms.ToTensor(),
            transforms.Normalize(mean=__mean, std=__std)] + ([manual_contrast] if config['manual_contrast'] else []))
        self.test_normalize = transforms.Compose([
            transforms.CenterCrop(util.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=__mean, std=__std)])

        td = {'strong': strong, 'simple': self.simple}
        self.aug = td[config['cspn_augmentation']]
        self.test_mode = False

    def __len__(self) -> int:
        return len(self.labeled_ds)

    def __getitem__(self, index: int):
        img, lb, _ = self.labeled_ds[index]
        if self.test_mode:
            img = self.test_normalize(img)
        else:
            img = self.aug(img)
        return img, lb, index