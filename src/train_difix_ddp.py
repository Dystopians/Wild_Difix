import os
import gc
import lpips
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision
import transformers
from torchvision.transforms.functional import crop
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, InitProcessGroupKwargs
from datetime import timedelta
from accelerate.utils import set_seed
from tqdm.auto import tqdm
from glob import glob
from einops import rearrange

import diffusers
from diffusers.utils.import_utils import is_xformers_available
from diffusers.optimization import get_scheduler

import wandb

from model_ddp import Difix
from transformers import AutoTokenizer, CLIPTextModel
from dataset import PairedDataset
from loss import gram_loss


def main(args):
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    init_pg_kwargs = InitProcessGroupKwargs(backend="gloo", timeout=timedelta(minutes=30))
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        kwargs_handlers=[ddp_kwargs, init_pg_kwargs],
    )

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "eval"), exist_ok=True)

    accelerator.print(f"[DDPDBG][rank={accelerator.process_index}] start building model")
    net_difix = Difix(
        lora_rank_vae=args.lora_rank_vae,
        timestep=args.timestep,
        mv_unet=args.mv_unet,
    )
    net_difix.set_train()
    accelerator.print(f"[DDPDBG][rank={accelerator.process_index}] model built")

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            net_difix.unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available, please install it by running `pip install xformers`")

    if args.gradient_checkpointing:
        net_difix.unet.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    accelerator.print(f"[DDPDBG][rank={accelerator.process_index}] build text enc + lpips/vgg")
    tokenizer = AutoTokenizer.from_pretrained("stabilityai/sd-turbo", subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained("stabilityai/sd-turbo", subfolder="text_encoder").eval().to(accelerator.device)
    net_lpips = lpips.LPIPS(net='vgg').cuda()
    net_lpips.requires_grad_(False)

    net_vgg = torchvision.models.vgg16(pretrained=True).features
    for param in net_vgg.parameters():
        param.requires_grad_(False)
    accelerator.print(f"[DDPDBG][rank={accelerator.process_index}] lpips/vgg ready")

    layers_to_opt = []
    layers_to_opt += list(net_difix.unet.parameters())
    for n, _p in net_difix.vae.named_parameters():
        if "lora" in n:
            assert _p.requires_grad
            layers_to_opt.append(_p)
    layers_to_opt = layers_to_opt + list(net_difix.vae.decoder.skip_conv_1.parameters()) + \
        list(net_difix.vae.decoder.skip_conv_2.parameters()) + \
        list(net_difix.vae.decoder.skip_conv_3.parameters()) + \
        list(net_difix.vae.decoder.skip_conv_4.parameters())

    optimizer = torch.optim.AdamW(layers_to_opt, lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,)
    lr_scheduler = get_scheduler(args.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles, power=args.lr_power,)

    accelerator.print(f"[DDPDBG][rank={accelerator.process_index}] build datasets")
    dataset_train = PairedDataset(dataset_path=args.dataset_path, split="train", tokenizer=tokenizer)
    dl_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers)
    dataset_val = PairedDataset(dataset_path=args.dataset_path, split="test", tokenizer=tokenizer)
    dl_val = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=0)
    accelerator.print(f"[DDPDBG][rank={accelerator.process_index}] datasets ready")

    global_step = 0

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # step-by-step prepare to locate stall position
    accelerator.print(f"[DDPDBG][rank={accelerator.process_index}] prepare begin (model)")
    net_difix = accelerator.prepare_model(net_difix)
    accelerator.print(f"[DDPDBG][rank={accelerator.process_index}] prepare model done")
    accelerator.print(f"[DDPDBG][rank={accelerator.process_index}] prepare begin (optimizer)")
    optimizer = accelerator.prepare_optimizer(optimizer)
    accelerator.print(f"[DDPDBG][rank={accelerator.process_index}] prepare optimizer done")
    accelerator.print(f"[DDPDBG][rank={accelerator.process_index}] prepare begin (dataloader)")
    dl_train = accelerator.prepare_data_loader(dl_train)
    accelerator.print(f"[DDPDBG][rank={accelerator.process_index}] prepare dataloader done")
    # lr_scheduler kept as-is
    net_lpips, net_vgg, text_encoder = accelerator.prepare(net_lpips, net_vgg, text_encoder)

    t_vgg_renorm =  torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    progress_bar = tqdm(range(0, args.max_train_steps), initial=global_step, desc="Steps",
        disable=not accelerator.is_local_main_process,)

    accelerator.print(f"[DDPDBG][rank={accelerator.process_index}] enter train loop")
    for epoch in range(0, args.num_training_epochs):
        for step, batch in enumerate(dl_train):
            if step == 0:
                accelerator.print(f"[DDPDBG][rank={accelerator.process_index}] got first batch")
            l_acc = [net_difix]
            with accelerator.accumulate(*l_acc):
                x_src = batch["conditioning_pixel_values"]
                x_tgt = batch["output_pixel_values"]
                B, V, C, H, W = x_src.shape
                # compute caption embeddings on device
                input_ids = batch.get("input_ids").to(accelerator.device)
                with torch.no_grad():
                    caption_enc = text_encoder(input_ids)[0]
                x_tgt_pred = net_difix(x_src, caption_enc=caption_enc)
                x_tgt = rearrange(x_tgt, 'b v c h w -> (b v) c h w')
                x_tgt_pred = rearrange(x_tgt_pred, 'b v c h w -> (b v) c h w')
                loss_l2 = F.mse_loss(x_tgt_pred.float(), x_tgt.float(), reduction="mean") * args.lambda_l2
                loss_lpips = net_lpips(x_tgt_pred.float(), x_tgt.float()).mean() * args.lambda_lpips
                loss = loss_l2 + loss_lpips
                if args.lambda_gram > 0:
                    if global_step > args.gram_loss_warmup_steps:
                        x_tgt_pred_renorm = t_vgg_renorm(x_tgt_pred * 0.5 + 0.5)
                        crop_h, crop_w = 400, 400
                        top, left = random.randint(0, H - crop_h), random.randint(0, W - crop_w)
                        x_tgt_pred_renorm = crop(x_tgt_pred_renorm, top, left, crop_h, crop_w)
                        x_tgt_renorm = t_vgg_renorm(x_tgt * 0.5 + 0.5)
                        x_tgt_renorm = crop(x_tgt_renorm, top, left, crop_h, crop_w)
                        loss_gram = gram_loss(x_tgt_pred_renorm.to(weight_dtype), x_tgt_renorm.to(weight_dtype), net_vgg) * args.lambda_gram
                        loss += loss_gram
                    else:
                        loss_gram = torch.tensor(0.0).to(weight_dtype)

                accelerator.backward(loss, retain_graph=False)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(layers_to_opt, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)
                x_tgt = rearrange(x_tgt, '(b v) c h w -> b v c h w', v=V)
                x_tgt_pred = rearrange(x_tgt_pred, '(b v) c h w -> b v c h w', v=V)

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambda_lpips", default=1.0, type=float)
    parser.add_argument("--lambda_l2", default=1.0, type=float)
    parser.add_argument("--lambda_gram", default=1.0, type=float)
    parser.add_argument("--gram_loss_warmup_steps", default=2000, type=int)
    parser.add_argument("--dataset_path", required=True, type=str)
    parser.add_argument("--prompt", default=None, type=str)
    parser.add_argument("--eval_freq", default=0, type=int)
    parser.add_argument("--num_samples_eval", type=int, default=0)
    parser.add_argument("--viz_freq", type=int, default=0)
    parser.add_argument("--tracker_project_name", type=str, default="difix")
    parser.add_argument("--tracker_run_name", type=str, default="difix_ddp")
    parser.add_argument("--pretrained_model_name_or_path")
    parser.add_argument("--revision", type=str, default=None,)
    parser.add_argument("--variant", type=str, default=None,)
    parser.add_argument("--tokenizer_name", type=str, default=None)
    parser.add_argument("--lora_rank_vae", default=4, type=int)
    parser.add_argument("--timestep", default=199, type=int)
    parser.add_argument("--mv_unet", action="store_true")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--cache_dir", default=None,)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resolution", type=int, default=512,)
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--num_training_epochs", type=int, default=10)
    parser.add_argument("--max_train_steps", type=int, default=100)
    parser.add_argument("--checkpointing_steps", type=int, default=500,)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true",)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--lr_num_cycles", type=int, default=1)
    parser.add_argument("--lr_power", type=float, default=1.0)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"],)
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true")
    parser.add_argument("--set_grads_to_none", action="store_true",)
    parser.add_argument("--resume", default=None, type=str)
    args = parser.parse_args()
    main(args)


