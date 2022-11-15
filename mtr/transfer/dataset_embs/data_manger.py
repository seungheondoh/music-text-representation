from torch.utils.data import DataLoader
from mtr.transfer.dataset_embs.mtat import MTAT_Dataset
from mtr.transfer.dataset_embs.gtzan import GTZAN_Dataset
from mtr.transfer.dataset_embs.fma import FMA_Dataset
from mtr.transfer.dataset_embs.kvt import KVT_Dataset
from mtr.transfer.dataset_embs.openmic import OPENMIC_Dataset
from mtr.transfer.dataset_embs.mtg import MTG_Dataset
from mtr.transfer.dataset_embs.emotify import EMOTIFY_Dataset

def get_dataloader(args, split, audio_embs):
    dataset = get_dataset(
        eval_dataset= args.eval_dataset,
        data_path= args.msu_dir,
        split= split,
        audio_embs= audio_embs
    )
    if split == "TRAIN":
        data_loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=False
        )
    elif split == "VALID":
        data_loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, drop_last=False
        )
    elif split == "TEST":
        data_loader = DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True, drop_last=False
        )
    elif split == "ALL":
        data_loader = DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True, drop_last=False
        )
    return data_loader


def get_dataset(
        eval_dataset,
        data_path,
        split,
        audio_embs
    ):
    if eval_dataset == "mtat":
        dataset = MTAT_Dataset(data_path, split, audio_embs)
    elif eval_dataset == "gtzan":
        dataset = GTZAN_Dataset(data_path, split, audio_embs)
    elif eval_dataset == "fma":
        dataset = FMA_Dataset(data_path, split, audio_embs)
    elif eval_dataset == "kvt":
        dataset = KVT_Dataset(data_path, split, audio_embs)
    elif eval_dataset == "openmic":
        dataset = OPENMIC_Dataset(data_path, split, audio_embs)
    elif eval_dataset == "emotify":
        dataset = EMOTIFY_Dataset(data_path, split, audio_embs)
    elif "mtg" in eval_dataset:
        dataset = MTG_Dataset(data_path, split, audio_embs, eval_dataset)
    else:
        print("error")
    return dataset