import torch
from torch.utils.data import DataLoader
from engine import Engine

qm9_to_eV = {'U0': 27.2114, 'U': 27.2114, 'G': 27.2114, 'H': 27.2114,
             'zpve': 27211.4, 'gap': 27.2114, 'homo': 27.2114, 'lumo': 27.2114}

for dataset in datasets.values():
    dataset.convert_units(qm9_to_eV)
    
dataloaders = {split: DataLoader(dataset,
                                 batch_size=args.batch_size,
                                 shuffle=args.shuffle if (split == 'train') else False,
                                 num_workers=args.num_workers,
                                 collate_fn=collate_fn)
               for split, dataset in datasets.items()}
