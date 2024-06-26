import torch
from torch.utils.data import random_split, DataLoader
import torchvision
import transforms as T

def get_dataloaders(hmdb_args):
    train_tfms = torchvision.transforms.Compose([
                                    T.ToFloatTensorInZeroOne(),
                                    T.Resize((128, 171)),
                                    T.RandomHorizontalFlip(),
                                    T.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]),
                                    T.RandomCrop((hmdb_args.crop_size, hmdb_args.crop_size))
                                ]) 

    test_tfms =  torchvision.transforms.Compose([
                                                T.ToFloatTensorInZeroOne(),
                                                T.Resize((128, 171)),
                                                T.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]),
                                                T.CenterCrop((hmdb_args.crop_size, hmdb_args.crop_size))
                                                ])

    hmdb51_train = torchvision.datasets.HMDB51(hmdb_args.video_dir, hmdb_args.split_dir, hmdb_args.num_frames,
                                                    step_between_clips = hmdb_args.clip_steps, fold=1, train=True,
                                                    transform=train_tfms, num_workers=hmdb_args.num_workers)

    hmdb51_test = torchvision.datasets.HMDB51(hmdb_args.video_dir, hmdb_args.split_dir, hmdb_args.num_frames,
                                                    step_between_clips = hmdb_args.clip_steps, fold=1, train=False,
                                                    transform=test_tfms, num_workers=hmdb_args.num_workers)
        
    total_train_samples = len(hmdb51_train)
    total_val_samples = round(hmdb_args.val_split * total_train_samples)

    print(f"number of train samples {total_train_samples}")
    print(f"number of validation samples {total_val_samples}")
    print(f"number of test samples {len(hmdb51_test)}")

    bs = hmdb_args.batch_size
    num_workers = hmdb_args.num_workers

    kwargs = {'num_workers':num_workers, 'pin_memory':True} if torch.cuda.is_available() else {'num_workers':num_workers}

    hmdb51_train_v1, hmdb51_val_v1 = random_split(hmdb51_train, [total_train_samples - total_val_samples,
                                                                        total_val_samples])
    
    train_loader = DataLoader(hmdb51_train_v1, batch_size=bs, shuffle=True, **kwargs)
    val_loader   = DataLoader(hmdb51_val_v1, batch_size=bs, shuffle=True, **kwargs) if hmdb_args.val_split > 0 else None
    test_loader  = DataLoader(hmdb51_test, batch_size=bs, shuffle=False, **kwargs)

    return train_loader, val_loader, test_loader