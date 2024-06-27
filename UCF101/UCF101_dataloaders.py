import torch
from torch.utils.data import random_split, DataLoader
import torchvision
import transforms as T

def custom_collate(batch):
    filtered_batch = []
    for video, _, label in batch:
        filtered_batch.append((video, label))
    return torch.utils.data.dataloader.default_collate(filtered_batch)

def get_dataloaders(ucf_args):
    train_tfms = torchvision.transforms.Compose([
                                    T.ToFloatTensorInZeroOne(),
                                    T.Resize((128, 171)),
                                    T.RandomHorizontalFlip(),
                                    T.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]),
                                    T.RandomCrop((ucf_args.crop_size, ucf_args.crop_size))
                                ]) 

    test_tfms =  torchvision.transforms.Compose([
                                                T.ToFloatTensorInZeroOne(),
                                                T.Resize((128, 171)),
                                                T.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]),
                                                T.CenterCrop((ucf_args.crop_size, ucf_args.crop_size))
                                                ])

    ucf101_train = torchvision.datasets.UCF101(ucf_args.video_dir, ucf_args.split_dir, ucf_args.num_frames,
                                                    step_between_clips = ucf_args.clip_steps, fold=1, train=True,
                                                    transform=train_tfms, num_workers=ucf_args.num_workers)

    ucf101_test = torchvision.datasets.UCF101(ucf_args.video_dir, ucf_args.split_dir, ucf_args.num_frames,
                                                    step_between_clips = ucf_args.clip_steps, fold=1, train=False,
                                                    transform=test_tfms, num_workers=ucf_args.num_workers)
        
    total_train_samples = len(ucf101_train)
    total_val_samples = round(ucf_args.val_split * total_train_samples)

    print(f"number of train samples {total_train_samples}")
    print(f"number of validation samples {total_val_samples}")
    print(f"number of test samples {len(ucf101_test)}")

    bs = ucf_args.batch_size
    num_workers = ucf_args.num_workers

    kwargs = {'num_workers':num_workers, 'pin_memory':True} if torch.cuda.is_available() else {'num_workers':num_workers}

    ucf101_train_v1, ucf101_val_v1 = random_split(ucf101_train, [total_train_samples - total_val_samples,
                                                                        total_val_samples])
    
    train_loader = DataLoader(ucf101_train_v1, batch_size=bs, shuffle=True, collate_fn=custom_collate, **kwargs)
    val_loader   = DataLoader(ucf101_val_v1, batch_size=bs, shuffle=True, collate_fn=custom_collate, **kwargs) if ucf_args.val_split > 0 else None
    test_loader  = DataLoader(ucf101_test, batch_size=bs, shuffle=False, collate_fn=custom_collate, **kwargs)

    return train_loader, val_loader, test_loader
