import os
import shutil
import tqdm
import torch

from opts import parser
from dataset import AdverbDataset

def main(args):
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    save_args(args)

    train_set = AdverbDataset(args.data_dir, args.feature_dir, agg=args.temporal_agg, modality=args.modality,
                                   window_size=args.t, adverb_filter=args.adverb_filter, phase='train')
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                              num_workers=args.workers)
    test_set = AdverbDataset(args.data_dir, args.feature_dir, agg=args.temporal_agg, modality=args.modality,
                                  window_size=args.t, adverb_filter=args.adverb_filter, phase='test')
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.workers)

    model = None

    start_epoch = 0
    test(model, test_loader, start_epoch)
    for epoch in range(start_epoch, start_epoch+args.max_epochs+1):
        train(model, train_loader, epoch)
        exit(0)

def train(model, train_loader, epoch):
    #model.train()
    for idx, data in tqdm.tqdm(enumerate(train_loader), total=len(train_loader)):
        print(data[0].shape)

def test(model, test_loader, epoch):
    for idx, data in tqdm.tqdm(enumerate(test_loader), total=len(test_loader)):
        print(data[0].shape)

def save_args(args):
    shutil.copy('train.py', args.checkpoint_dir)
    #shutil.copy('models.py', args.checkpoint_dir)
    with open(os.path.join(args.checkpoint_dir, 'args.txt'), 'w') as f:
        f.write(str(args))

if __name__ == '__main__':
    args = parser.parse_args()
    if args.modality == 'both':
        args.modality = ['rgb', 'flow']
    else:
        args.modality = [args.modality]
    main(args)
