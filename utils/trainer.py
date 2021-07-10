

import torch
import torch.nn.utils as utils
from utils.utils import timer,quantize,make_optimizer
from utils.psnr import calc_psnr
class Trainer():
    def __init__(self, args, loader, my_model):
        self.args = args
        self.loader_train , self.loader_eval = loader
        self.model = my_model
        if self.args.loss == "mse":
            self.loss = torch.nn.MSELoss()
        elif self.args.loss == "l1":
            self.loss = torch.nn.L1Loss()
        self.optimizer = make_optimizer(args, self.model)
        self.scaler = torch.cuda.amp.GradScaler() 
        self.precision = args.precision

        self.error_last = 1e8
        self.train_loss = []
        self.eval_loss = []
        self.eval_psnr = []

    def train(self):
        epoch = self.optimizer.get_last_epoch() + 1
        lr = self.optimizer.get_lr()
        self.model.train()
        timer_data, timer_model = timer(), timer()
        loss_value = 0
        for batch, (input,target) in enumerate(self.loader_train):
            input,target = self.prepare(input,target)
            timer_data.hold()
            timer_model.tic()
            self.optimizer.zero_grad()
            if self.precision == 'amp':
                with torch.cuda.amp.autocast(): 
                    input = self.model(input)
            else:
                input = self.model(input)
            loss = self.loss(input, target)
            if self.precision == 'amp':
                self.scaler.scale(loss).backward() 
                self.scaler.step(self.optimizer)
                self.scaler.update() 
            else:
                self.optimizer.step()
            loss_value += loss.cpu().detach().item()
            self.train_loss.append(loss.cpu().detach().item())

            timer_model.hold()
            if (batch + 1) % self.args.print_every == 0:
                print('Epoch [{}/{}]\tBatch[{}/{}]\tLoss {:.4f}\tlr {:.4f}\tTime {:.1f}+{:.1f}s'.format(
                    epoch,
                    self.args.epochs,
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    loss_value/(batch + 1) ,
                    lr,
                    timer_model.release(),
                    timer_data.release()),
                )
            timer_data.tic()
        self.optimizer.schedule()
    def eval(self):
        torch.set_grad_enabled(False)
        self.model.eval()
        avg_psnr = 0
        best_psnr = 0
        loss_value = 0
        for idx_data, (lr, hr) in enumerate(self.loader_eval):

            lr, hr = self.prepare(lr, hr)
            sr = self.model(lr)
            sr = quantize(sr, 255)
            loss = self.loss(lr,sr)
            loss_value += loss.cpu().detach().item()
            self.eval_loss.append(loss.cpu().detach().item())
            psnr = calc_psnr(sr, hr , 255)
            self.eval_psnr.append(psnr)
            if best_psnr<psnr:
                best_psnr = psnr
            avg_psnr +=psnr
        avg_psnr /= idx_data
        loss_value /= idx_data
        print(f'AVG PSNR {avg_psnr:.4f}\tBest PSNR {best_psnr:.4f}\tLoss {loss_value:.4f}')
        torch.set_grad_enabled(True)
    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def terminate(self):

        epoch = self.optimizer.get_last_epoch() + 1
        return epoch >= self.args.epochs