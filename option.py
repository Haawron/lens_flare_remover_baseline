import argparse

parser = argparse.ArgumentParser(description='For Dacon ')

parser.add_argument('--debug', action='store_true',
                    help='Enables debug mode')
parser.add_argument('--train', action='store_false',
                    help='Enables train mode')

# Hardware specifications
parser.add_argument('--n_threads', type=int, default=12,
                    help='number of threads for data loading')
parser.add_argument('--cpu', action='store_true',
                    help='use cpu only')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')

parser.add_argument('--dir_data', type=str, default='dataset',
                    help='dataset directory')
parser.add_argument('--data_size', type=int, default=622,
                    help='dataset directory')
parser.add_argument('--patch_size', type=int, default=128,
                    help='output patch size')

# Model specifications
parser.add_argument('--model', default='baseline',
                    help='model name')
parser.add_argument('--dropout_rate', type=float, default=.1,
                    help='dropout rate')
parser.add_argument('--precision', type=str, default='single',
                    choices=('single', 'amp'),
                    help='FP precision for test (single | amp)')

# Training specifications

parser.add_argument('--test_every', type=int, default=1000,
                    help='do test per every N batches')
parser.add_argument('--epochs', type=int, default=60,
                    help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=2,
                    help='input batch size for training')



# Optimization specifications
parser.add_argument('--lr', type=float, default=1e-3,
                    help='learning rate')
parser.add_argument('--decay', type=str, default='50',
                    help='learning rate decay type')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='learning rate decay factor for step decay')
parser.add_argument('--optimizer', default='ADAM',
                    choices=('SGD', 'ADAM', 'RMSprop'),
                    help='optimizer to use (SGD | ADAM | RMSprop)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--betas', type=tuple, default=(0.9, 0.999),
                    help='ADAM beta')
parser.add_argument('--epsilon', type=float, default=1e-8,
                    help='ADAM epsilon for numerical stability')
parser.add_argument('--weight_decay', type=float, default=1e-4,
                    help='weight decay')

# Loss specifications
parser.add_argument('--loss', type=str, default='l1',
                    help='loss function configuration')

# Log specifications
parser.add_argument('--save', type=str, default='test',
                    help='file name to save')
parser.add_argument('--load', type=str, default='',
                    help='file name to load')
parser.add_argument('--save_models', action='store_true',
                    help='save all intermediate models')
parser.add_argument('--print_every', type=int, default=100,
                    help='how many batches to wait before logging training status')
parser.add_argument('--save_results', action='store_true',
                    help='save output results')


args = parser.parse_args()

if args.epochs == 0:
    args.epochs = 1e8

for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False