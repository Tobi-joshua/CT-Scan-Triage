from __future__ import annotations
from pathlib import Path
from torchvision import datasets, transforms

IMAGENET_MEAN=(0.485,0.456,0.406)
IMAGENET_STD=(0.229,0.224,0.225)

def transforms_for(size=224):
    train = transforms.Compose([
        transforms.Resize((size,size)),
        transforms.RandomRotation(7),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=.1, contrast=.1),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    valid = transforms.Compose([
        transforms.Resize((size,size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train, valid

def imagefolder_loaders(root, batch_size=32, size=224, workers=2):
    root=Path(root)
    tr, va=transforms_for(size)
    train_ds=datasets.ImageFolder(root/"train", transform=tr)
    val_ds=datasets.ImageFolder(root/"val", transform=va)
    test_ds=datasets.ImageFolder(root/"test", transform=va) if (root/"test").exists() else None
    train_loader=__import__("torch").utils.data.DataLoader(train_ds,batch_size=batch_size,shuffle=True,num_workers=workers)
    val_loader=__import__("torch").utils.data.DataLoader(val_ds,batch_size=batch_size,shuffle=False,num_workers=workers)
    test_loader=None if test_ds is None else __import__("torch").utils.data.DataLoader(test_ds,batch_size=batch_size,shuffle=False,num_workers=workers)
    return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader
