import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', default='data/',
                    help='directory containing the training test splits and list of antonyms')
parser.add_argument('--feature-dir', default='features/', help='directory containing the rgb and flow features')
parser.add_argument('--checkpoint-dir', default='tmp/', help='directory to save checkpoints and tensorflow logs to')
parser.add_argument('--load', default=None, help='path to checkpoint to load')
parser.add_argument('--adverb-filter', nargs='+', default=None, help='select adverbs to train')

## model parameters
parser.add_argument('--emb-dim', type=int, default=300, help='dimension of common embedding space')
parser.add_argument('--no-glove-init', dest='glove-init', action='store_false',
                    help='don\'t initialize the action embeddings with word vectors')
parser.add_argument('--temporal-agg', default='sdp', choices=['single', 'average', 'sdp'],
                    help='method to aggregate the features in the window of size T')
parser.add_argument('--modality', default='both', choices=['rgb', 'flow', 'both'],
                    help='modalities used to train the model')
parser.add_argument('--t', type=int, default=20, help='size of the temporal window used around the weak timestamp')

## optimization
parser.add_argument('--batch-size', type=int, default=512, help='number of data points in a batch')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--wd', type=float, default=5e-5, help='weight decay')
parser.add_argument('--workers', type=int, default=8, help='number of workers used to load data')
parser.add_argument('--save-interval', type=int, default=100, help='number of epochs to save a checkpoint after')
parser.add_argument('--eval-interval', type=int, default=20, help='number of epochs to test after')
parser.add_argument('--max-epochs', type=int, default=1000, help='max_epochs to run the model for')
parser.add_argument('--no-gpu', dest='gpu', action='store_false', help='run only on the cpu')



