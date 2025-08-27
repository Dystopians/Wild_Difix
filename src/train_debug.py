import os
import time
import argparse
import torch
from accelerate import Accelerator
from dataset import PairedDataset


def main(args):
    print("[DBG] stage=init")
    accelerator = Accelerator(log_with=None)
    print("[DBG] stage=accelerator_created")

    ds = PairedDataset(args.dataset_path, "train", height=512, width=512, tokenizer=None)
    print("[DBG] stage=dataset_loaded len=", len(ds))

    # Try plain DataLoader CPU only
    t = time.time()
    item = ds[0]
    print("[DBG] stage=getitem0 dt=", round(time.time()-t,3))

    dl = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=True, num_workers=args.num_workers)
    print("[DBG] stage=dataloader_created workers=", args.num_workers)

    it = iter(dl)
    t = time.time()
    batch = next(it)
    print("[DBG] stage=first_batch dt=", round(time.time()-t,3))
    for k, v in batch.items():
        if hasattr(v, "shape"):
            print("[DBG] batch[{}].shape = {}".format(k, tuple(v.shape)))
        else:
            print("[DBG] batch[{}] type = {}".format(k, type(v)))

    # Try accelerator.prepare only to see if it stalls there
    net = torch.nn.Identity()
    print("[DBG] stage=model_created")
    net, dl = accelerator.prepare(net, dl)
    print("[DBG] stage=accelerator_prepared")

    it = iter(dl)
    t = time.time()
    batch = next(it)
    print("[DBG] stage=first_batch_after_prepare dt=", round(time.time()-t,3))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()
    main(args)


