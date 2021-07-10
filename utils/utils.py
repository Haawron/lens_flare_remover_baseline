import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
import time
import random
from PIL import Image
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14,'font.family':'Arial'})
def image_transform(input,target):
    transform = transforms.Compose([
        transforms.AutoAugment(),
        transforms.ToTensor(),
    ])
    crop = transforms.RandomCrop(224)
    img_mask = crop.get_params(input, (224,224))
    img_mask = np.array(img_mask)
    img_mask = np.concatenate((img_mask[:2], img_mask[:2] + img_mask[2:]))
    input = input.crop(img_mask)
    target = target.crop(img_mask)

    return transform(input),transform(target)


def image_transform_dacon(input, target):
    input= cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
    target= cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
    input=Image.fromarray(input)
    target=Image.fromarray(target)
    angle = random.choice([0, 90, 180, 270])
    input = TF.rotate(input, angle)
    target = TF.rotate(target, angle)

    if random.random() < .5:
        input = TF.hflip(input)
        target = TF.hflip(target)
        
    transform = [transforms.ToTensor()]
    transform = transforms.Compose(transform)
    return transform(input), transform(target)


def make_optimizer(args, target):
    '''
        make optimizer and scheduler together
    '''
    # optimizer
    trainable = filter(lambda x: x.requires_grad, target.parameters())
    kwargs_optimizer = {'lr': args.lr, 'weight_decay': args.weight_decay}

    if args.optimizer == 'SGD':
        optimizer_class = optim.SGD
        kwargs_optimizer['momentum'] = args.momentum
    elif args.optimizer == 'ADAM':
        optimizer_class = optim.Adam
        kwargs_optimizer['betas'] = args.betas
        kwargs_optimizer['eps'] = args.epsilon
    elif args.optimizer == 'RMSprop':
        optimizer_class = optim.RMSprop
        kwargs_optimizer['eps'] = args.epsilon

    # scheduler
    milestones = list(map(lambda x: int(x), args.decay.split('-')))
    kwargs_scheduler = {'T_max':args.epochs} #{'milestones': milestones, 'gamma': args.gamma}
    scheduler_class = lrs.CosineAnnealingLR

    class CustomOptimizer(optimizer_class):
        def __init__(self, *args, **kwargs):
            super(CustomOptimizer, self).__init__(*args, **kwargs)

        def _register_scheduler(self, scheduler_class, **kwargs):
            self.scheduler = scheduler_class(self, **kwargs)

        def save(self, save_dir):
            torch.save(self.state_dict(), self.get_dir(save_dir))

        def load(self, load_dir, epoch=1):
            self.load_state_dict(torch.load(self.get_dir(load_dir)))
            if epoch > 1:
                for _ in range(epoch): self.scheduler.step()



        def schedule(self):
            self.scheduler.step()

        def get_lr(self):
            return self.scheduler.get_last_lr()[0]

        def get_last_epoch(self):
            return self.scheduler.last_epoch

    optimizer = CustomOptimizer(trainable, **kwargs_optimizer)
    optimizer._register_scheduler(scheduler_class, **kwargs_scheduler)
    return optimizer



class timer():
    def __init__(self):
        self.acc = 0
        self.tic()

    def tic(self):
        self.t0 = time.time()

    def toc(self, restart=False):
        diff = time.time() - self.t0
        if restart: self.t0 = time.time()
        return diff

    def hold(self):
        self.acc += self.toc()

    def release(self):
        ret = self.acc
        self.acc = 0

        return ret

    def reset(self):
        self.acc = 0
def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)

def plot_loss( epoch,loss,save_dir):
    axis = np.linspace(1, epoch, epoch)
    train_loss , eval_loss = loss
    fig = plt.figure(figsize=(8,8))
    plt.title("PSNR")
    plt.plot(np.arange(epoch) +1,train_loss,label='train')
    plt.plot(np.arange(epoch) + 1, eval_loss, label='valid')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('PSNR')
    plt.grid(True)
    plt.savefig(f"{save_dir}/loss.png")
    plt.close(fig)
def plot_psnr( epoch,psnr,save_dir):
    axis = np.linspace(1, epoch, epoch)
    fig = plt.figure(figsize=(8,8))
    plt.title("PSNR")
    plt.plot(np.arange(epoch)+1,psnr)
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('PSNR')
    plt.grid(True)
    plt.savefig(f"{save_dir}/psnr.png")
    plt.close(fig)
