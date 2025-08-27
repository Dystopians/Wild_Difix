import argparse
from accelerate import Accelerator
import torch
from model import Difix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mv_unet", action="store_true")
    args = parser.parse_args()

    acc = Accelerator()
    print(f"[PDBG][rank={acc.process_index}] init")

    net = Difix(mv_unet=args.mv_unet)
    num_params = sum(1 for _ in net.parameters())
    num_trainable = sum(1 for p in net.parameters() if p.requires_grad)
    print(f"[PDBG][rank={acc.process_index}] before_prepare: params={num_params} trainable={num_trainable}")

    try:
        net_prep, = acc.prepare(net)
        print(f"[PDBG][rank={acc.process_index}] after_prepare ok")
    except Exception as e:
        print(f"[PDBG][rank={acc.process_index}] after_prepare EXC: {e}")


if __name__ == "__main__":
    main()


