import argparse

from torch.nn import L1Loss

import torch.optim as opt
from torch_geometric.loader import DataLoader

from model.refnet import RefNetWrapper
from trainer import Trainer

def main(args):
    dataset = QM9(root="dataset/", transform=None, pre_transform=None, pre_filter=None)
    train_size = 110000
    valid_size = 10000
    seed = 42
    split_idx = dataset.get_idx_split(len(dataset), train_size=train_size, valid_size=valid_size,
                                           seed=seed)
    train_data, val_data, test_data = dataset[split_idx['train']], dataset[
        split_idx['valid']], dataset[split_idx['test']]
    train, valid, test = dataset[0], dataset[1], dataset[2]

    mean = train_data.mean()
    std = train_data.std()

    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
    test_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    model = RefNetWrapper(
        hidden_dim=128,
        num_layers=8,
        num_rbf=20,
        cutoff=5.0,
        r_basis="BesselBasis",
        activation="swish",
        max_z=100,
        weight_init="xavier_uniform",
        bias_init="zeros",
        vector_embed_dropout=0.0,
        target_mean=mean[args.target],
        target_std=std[args.target],
        decoder=None,
    )
    # create optimizer
    optimizer = opt.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        eps=1e-7,
    )

    scheduler = opt.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=args.lr_decay,
        patience=args.lr_patience,
        threshold=1e-6,
        cooldown=args.lr_patience // 2,
        min_lr=1e-6,
    )


    trainer = Trainer(model=model, optimizer=optimizer, loss_fn=L1Loss(), device="cuda", label_fn=lambda batch: batch.y[args.target], scheduler=scheduler)
    trainer.fit(train_dataloader, val_dataloader, epochs=args.epochs)
    trainer.test(test_dataloader)


# check name is main
if __name__ == '__main__':
    # argument parser
    parser = argparse.ArgumentParser(description='RefNet training script on QM9 dataset')
    parser.add_argument('--target', default="alpha")
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--lr_decay', type=float, default=0.1)
    parser.add_argument('--lr_patience', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=1000)

    # parse args
    args = parser.parse_args()
    main(args)

