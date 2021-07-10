import os
import json
import logging
import torch
import model
from  utils import dataset,trainer
from option import args
from torch.utils.data.dataloader import DataLoader
from  utils.utils import plot_loss,plot_psnr
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
torch.manual_seed(args.seed)
torch.backends.cudnn.benchmark = True

def main():
    save_dir = f'{args.save}/{args.model}'
    os.makedirs(save_dir, exist_ok=True)
    _dataset = dataset.Dataset(args)
    len_dataset = len(_dataset)
    train_set, test_set = torch.utils.data.random_split(_dataset, [round(len_dataset * .8),
                                                                  len_dataset - round(len_dataset * .8)])
    _loader = [DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=not args.cpu,
        num_workers=args.n_threads,
         drop_last = True,
    ),
        DataLoader(
            test_set,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=not args.cpu,
            num_workers=args.n_threads,
            drop_last=True,
        ),
        ]
    _model = model.Model(args)
    t = trainer.Trainer(args, _loader, _model)

    while not t.terminate():
        t.train()
        t.eval()

    with open(os.path.join(f'{save_dir}', 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    plot_loss(args.epochs,[t.train_loss,t.eval_loss],save_dir)
    plot_psnr(args.epochs, t.eval_psnr, save_dir)
    torch.save(model.state_dict(),f"{save_dir}/model.pth")

if __name__ == '__main__':
    main()
