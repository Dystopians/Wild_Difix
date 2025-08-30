import os
import gc
import lpips
import random
import argparse
import numpy as np
import math
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision
import transformers
from torchvision.transforms.functional import crop
from accelerate import Accelerator
from accelerate.utils import set_seed, DistributedDataParallelKwargs
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from glob import glob
from einops import rearrange

import diffusers
from diffusers.utils.import_utils import is_xformers_available
from diffusers.optimization import get_scheduler

import wandb

from model import Difix, load_ckpt_from_state_dict, save_ckpt
from dataset import PairedDataset
from loss import gram_loss


def main(args):
    ddp_kwargs = DistributedDataParallelKwargs(
        broadcast_buffers=False,
        find_unused_parameters=False,
        static_graph=False,
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        kwargs_handlers=[ddp_kwargs],
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

    net_difix = Difix(
        lora_rank_vae=args.lora_rank_vae, 
        timestep=args.timestep,
        mv_unet=args.mv_unet,
        freeze_unet=args.freeze_unet,
        train_unet=args.train_unet,
    )
    net_difix.set_train()

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            net_difix.unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available, please install it by running `pip install xformers`")

    if args.gradient_checkpointing:
        net_difix.unet.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    net_lpips = lpips.LPIPS(net=args.lpips_net)

    net_lpips.requires_grad_(False)
    
    net_vgg = torchvision.models.vgg16(pretrained=True).features.float()
    for param in net_vgg.parameters():
        param.requires_grad_(False)

    # make the optimizer
    layers_to_opt = []
    if not args.freeze_unet and args.train_unet:
        layers_to_opt += list(net_difix.unet.parameters())
   
    for n, _p in net_difix.vae.named_parameters():
        if "lora" in n and "vae_skip" in n:
            assert _p.requires_grad
            layers_to_opt.append(_p)
    layers_to_opt = layers_to_opt + list(net_difix.vae.decoder.skip_conv_1.parameters()) + \
        list(net_difix.vae.decoder.skip_conv_2.parameters()) + \
        list(net_difix.vae.decoder.skip_conv_3.parameters()) + \
        list(net_difix.vae.decoder.skip_conv_4.parameters())

    # Deduplicate parameters to avoid duplicate params across groups
    _seen, _unique = set(), []
    for p in layers_to_opt:
        _id = id(p)
        if _id not in _seen:
            _unique.append(p)
            _seen.add(_id)
    if args.use_8bit_optimizer:
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(_unique, lr=args.learning_rate,
                betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
                eps=args.adam_epsilon)
        except Exception as e:
            print(f"bitsandbytes not available ({e}), falling back to torch AdamW")
            optimizer = torch.optim.AdamW(_unique, lr=args.learning_rate,
                betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
                eps=args.adam_epsilon,)
    else:
        optimizer = torch.optim.AdamW(_unique, lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,)
    dataset_train = PairedDataset(dataset_path=args.dataset_path, split="train", tokenizer=net_difix.tokenizer)
    dl_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers)
    # Build LR scheduler after computing total target steps (now dl_train is defined)
    steps_per_epoch = math.ceil(len(dl_train) / max(1, args.gradient_accumulation_steps)) if len(dl_train) > 0 else 1
    target_max_steps = args.max_train_steps if args.max_train_steps and args.max_train_steps > 0 else steps_per_epoch * max(1, args.num_training_epochs)
    lr_scheduler = get_scheduler(args.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=min(args.lr_warmup_steps, target_max_steps),
        num_training_steps=target_max_steps,
        num_cycles=args.lr_num_cycles, power=args.lr_power,)
    dataset_val = PairedDataset(dataset_path=args.dataset_path, split="test", tokenizer=net_difix.tokenizer)
    random.Random(42).shuffle(dataset_val.img_names)
    dl_val = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=0)

    # Resume from checkpoint
    global_step = 0    
    if args.resume is not None:
        if os.path.isdir(args.resume):
            # Resume from last ckpt
            ckpt_files = glob(os.path.join(args.resume, "*.pkl"))
            assert len(ckpt_files) > 0, f"No checkpoint files found: {args.resume}"
            ckpt_files = sorted(ckpt_files, key=lambda x: int(x.split("/")[-1].replace("model_", "").replace(".pkl", "")))
            print("="*50); print(f"Loading checkpoint from {ckpt_files[-1]}"); print("="*50)
            global_step = int(ckpt_files[-1].split("/")[-1].replace("model_", "").replace(".pkl", ""))
            net_difix, optimizer = load_ckpt_from_state_dict(
                net_difix, optimizer, ckpt_files[-1]
            )
        elif args.resume.endswith(".pkl"):
            print("="*50); print(f"Loading checkpoint from {args.resume}"); print("="*50)
            global_step = int(args.resume.split("/")[-1].replace("model_", "").replace(".pkl", ""))
            net_difix, optimizer = load_ckpt_from_state_dict(
                net_difix, optimizer, args.resume
            )    
        else:
            raise NotImplementedError(f"Invalid resume path: {args.resume}")
    else:
        print("="*50); print(f"Training from scratch"); print("="*50)
    
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Quick signature check before DDP: all ranks must match
    def _param_signature(m):
        return tuple([tuple(p.shape) for p in m.parameters() if p.requires_grad])
    sig = _param_signature(net_difix)
    print(f"[rank={accelerator.process_index}] trainable_params={len(sig)} first3={sig[:3]}", flush=True)
    accelerator.wait_for_everyone()

    # One-shot prepare: place models/optimizer/dataloaders/devices via Accelerator
    # Keep VGG out of prepare so it stays float32 and is not auto-casted to bf16
    net_difix, net_lpips, optimizer, dl_train, lr_scheduler = accelerator.prepare(
        net_difix, net_lpips, optimizer, dl_train, lr_scheduler
    )
    # Place VGG on the correct device with float32
    net_vgg = net_vgg.to(accelerator.device, dtype=torch.float32)
    # renorm with image net statistics
    t_vgg_renorm =  transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        init_kwargs = {
            "wandb": {
                "name": args.tracker_run_name,
                "dir": args.output_dir,
            },
        }        
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config, init_kwargs=init_kwargs)

    progress_bar = tqdm(range(0, target_max_steps), initial=global_step, desc="Steps",
        disable=not accelerator.is_local_main_process,)

    # start the training loop
    should_break = False
    for epoch in range(0, args.num_training_epochs):
        for step, batch in enumerate(dl_train):
            l_acc = [net_difix]
            with accelerator.accumulate(*l_acc):
                x_src = batch["conditioning_pixel_values"]
                x_tgt = batch["output_pixel_values"]
                B, V, C, H, W = x_src.shape

                # Optionally micro-batch over views to reduce memory
                total_loss = 0.0
                l2_total, lpips_total, gram_total = 0.0, 0.0, 0.0
                V_mb = args.views_per_microbatch
                num_chunks = (V + V_mb - 1) // V_mb
                for v0 in range(0, V, V_mb):
                    x_src_mb = x_src[:, v0:v0+V_mb]
                    x_tgt_mb = x_tgt[:, v0:v0+V_mb]

                    x_tgt_pred_mb = net_difix(x_src_mb, prompt_tokens=batch["input_ids"])       
                    
                    xt   = rearrange(x_tgt_mb, 'b v c h w -> (b v) c h w')
                    xt_p = rearrange(x_tgt_pred_mb, 'b v c h w -> (b v) c h w')
                             
                    # Reconstruction loss
                    loss_l2_mb = F.mse_loss(xt_p.float(), xt.float(), reduction="mean") * args.lambda_l2

                    # LPIPS with optional downsample and view cap
                    x_lpips_pred = xt_p
                    x_lpips_tgt  = xt
                    if args.lpips_downsample is not None and args.lpips_downsample > 0 and (H != args.lpips_downsample or W != args.lpips_downsample):
                        ds = args.lpips_downsample
                        x_lpips_pred = F.interpolate(x_lpips_pred, size=(ds, ds), mode="bilinear", align_corners=False)
                        x_lpips_tgt  = F.interpolate(x_lpips_tgt,  size=(ds, ds), mode="bilinear", align_corners=False)
                    loss_lpips_mb = net_lpips(x_lpips_pred.float(), x_lpips_tgt.float()).mean() * args.lambda_lpips

                    loss_mb = loss_l2_mb + loss_lpips_mb
                    # Gram matrix loss (compute on full-res crop)
                    if args.lambda_gram > 0:
                        if global_step > args.gram_loss_warmup_steps:
                            # Disable autocast to keep VGG path strictly in float32 and avoid bf16/float bias mismatch
                            with torch.amp.autocast('cuda', enabled=False):
                                x_tgt_pred_renorm = t_vgg_renorm(xt_p.float() * 0.5 + 0.5).float()
                                crop_h, crop_w = args.gram_crop_size, args.gram_crop_size
                                top, left = random.randint(0, H - crop_h), random.randint(0, W - crop_w)
                                x_tgt_pred_renorm = crop(x_tgt_pred_renorm, top, left, crop_h, crop_w).float()
                                x_tgt_renorm = t_vgg_renorm(xt.float() * 0.5 + 0.5).float()
                                x_tgt_renorm = crop(x_tgt_renorm, top, left, crop_h, crop_w).float()
                                loss_gram_mb = gram_loss(x_tgt_pred_renorm, x_tgt_renorm, net_vgg.float()) * args.lambda_gram
                            loss_mb = loss_mb + loss_gram_mb
                            gram_total += loss_gram_mb.detach().item()
                        else:
                            pass

                    accelerator.backward(loss_mb / num_chunks, retain_graph=False)

                    l2_total += loss_l2_mb.detach().item()
                    lpips_total += loss_lpips_mb.detach().item()
                    total_loss += loss_mb.item()

                # expose last micro-batch tensors for optional visualization
                x_tgt_pred = x_tgt_pred_mb
                x_tgt = x_tgt_mb

                # collected per-microbatch grads above; no second backward here
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(layers_to_opt, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)
                
                # No need to reshape back for logging unless visualizing

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    logs = {}
                    # log averaged micro-batch losses
                    logs["loss_l2"] = l2_total / max(num_chunks, 1)
                    logs["loss_lpips"] = lpips_total / max(num_chunks, 1)
                    if args.lambda_gram > 0:
                        logs["loss_gram"] = gram_total / max(num_chunks, 1)
                    progress_bar.set_postfix(**logs)

                    # viz some images
                    if global_step % args.viz_freq == 1:
                        log_dict = {
                            "train/source": [wandb.Image((rearrange(x_src, "b v c h w -> b c (v h) w")[idx].float().detach().cpu() * 0.5 + 0.5).clamp(0,1), caption=f"idx={idx}") for idx in range(B)],
                            "train/target": [wandb.Image((rearrange(x_tgt, "b v c h w -> b c (v h) w")[idx].float().detach().cpu() * 0.5 + 0.5).clamp(0,1), caption=f"idx={idx}") for idx in range(B)],
                            "train/model_output": [wandb.Image((rearrange(x_tgt_pred, "b v c h w -> b c (v h) w")[idx].float().detach().cpu() * 0.5 + 0.5).clamp(0,1), caption=f"idx={idx}") for idx in range(B)],
                        }
                        for k in log_dict:
                            logs[k] = log_dict[k]

                    # checkpoint the model
                    if global_step == 1 or (args.checkpointing_steps > 0 and global_step % args.checkpointing_steps == 0):
                        outf = os.path.join(args.output_dir, "checkpoints", f"model_{global_step}.pkl")
                        # accelerator.unwrap_model(net_difix).save_model(outf)
                        save_ckpt(accelerator.unwrap_model(net_difix), optimizer, outf)

                    # compute validation set L2, LPIPS
                    if args.eval_freq > 0 and global_step % args.eval_freq == 1:
                        l_l2, l_lpips = [], []
                        log_dict = {"sample/source": [], "sample/target": [], "sample/model_output": []}
                        for step, batch_val in enumerate(dl_val):
                            if step >= args.num_samples_eval:
                                break
                            x_src = batch_val["conditioning_pixel_values"].to(accelerator.device, dtype=weight_dtype)
                            x_tgt = batch_val["output_pixel_values"].to(accelerator.device, dtype=weight_dtype)
                            B, V, C, H, W = x_src.shape
                            assert B == 1, "Use batch size 1 for eval."
                            with torch.no_grad():
                                # forward pass
                                x_tgt_pred = accelerator.unwrap_model(net_difix)(
                                    x_src, prompt_tokens=batch_val["input_ids"].to(accelerator.device)
                                )
                                
                                if step % 10 == 0:
                                    log_dict["sample/source"].append(wandb.Image(rearrange(x_src, "b v c h w -> b c (v h) w")[0].float().detach().cpu(), caption=f"idx={len(log_dict['sample/source'])}"))
                                    log_dict["sample/target"].append(wandb.Image(rearrange(x_tgt, "b v c h w -> b c (v h) w")[0].float().detach().cpu(), caption=f"idx={len(log_dict['sample/source'])}"))
                                    log_dict["sample/model_output"].append(wandb.Image(rearrange(x_tgt_pred, "b v c h w -> b c (v h) w")[0].float().detach().cpu(), caption=f"idx={len(log_dict['sample/source'])}"))
                                
                                x_tgt = x_tgt[:, 0] # take the input view
                                x_tgt_pred = x_tgt_pred[:, 0] # take the input view
                                # compute the reconstruction losses
                                loss_l2 = F.mse_loss(x_tgt_pred.float(), x_tgt.float(), reduction="mean")
                                loss_lpips = net_lpips(x_tgt_pred.float(), x_tgt.float()).mean()

                                l_l2.append(loss_l2.item())
                                l_lpips.append(loss_lpips.item())

                        logs["val/l2"] = np.mean(l_l2)
                        logs["val/lpips"] = np.mean(l_lpips)
                        for k in log_dict:
                            logs[k] = log_dict[k]
                        gc.collect()
                        torch.cuda.empty_cache()
                    accelerator.log(logs, step=global_step)
                if global_step >= target_max_steps:
                    should_break = True
                    break
        if should_break:
            break


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    # args for the loss function
    parser.add_argument("--lambda_lpips", default=1.0, type=float)
    parser.add_argument("--lambda_l2", default=1.0, type=float)
    parser.add_argument("--lambda_gram", default=1.0, type=float)
    parser.add_argument("--gram_loss_warmup_steps", default=2000, type=int)

    # dataset options
    parser.add_argument("--dataset_path", required=True, type=str)
    parser.add_argument("--train_image_prep", default="resized_crop_512", type=str)
    parser.add_argument("--test_image_prep", default="resized_crop_512", type=str)
    parser.add_argument("--prompt", default=None, type=str)

    # validation eval args
    parser.add_argument("--eval_freq", default=100, type=int)
    parser.add_argument("--num_samples_eval", type=int, default=100, help="Number of samples to use for all evaluation")

    parser.add_argument("--viz_freq", type=int, default=100, help="Frequency of visualizing the outputs.")
    parser.add_argument("--tracker_project_name", type=str, default="difix", help="The name of the wandb project to log to.")
    parser.add_argument("--tracker_run_name", type=str, required=True)

    # details about the model architecture
    parser.add_argument("--pretrained_model_name_or_path")
    parser.add_argument("--revision", type=str, default=None,)
    parser.add_argument("--variant", type=str, default=None,)
    parser.add_argument("--tokenizer_name", type=str, default=None)
    parser.add_argument("--lora_rank_vae", default=4, type=int)
    parser.add_argument("--timestep", default=199, type=int)
    parser.add_argument("--mv_unet", action="store_true")

    # training details
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--cache_dir", default=None,)
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--resolution", type=int, default=512,)
    parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--num_training_epochs", type=int, default=10)
    parser.add_argument("--max_train_steps", type=int, default=10_000,)
    parser.add_argument("--checkpointing_steps", type=int, default=500,)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.",)
    parser.add_argument("--gradient_checkpointing", action="store_true",)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--lr_scheduler", type=str, default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--lr_num_cycles", type=int, default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")

    parser.add_argument("--dataloader_num_workers", type=int, default=0,)
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--allow_tf32", action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--report_to", type=str, default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"],)
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers.")
    parser.add_argument("--set_grads_to_none", action="store_true",)
    parser.add_argument("--use_8bit_optimizer", action="store_true")
    parser.add_argument("--views_per_microbatch", type=int, default=1)
    parser.add_argument("--train_unet", action="store_true")
    parser.add_argument("--freeze_unet", action="store_true")
    parser.add_argument("--gram_crop_size", default=256, type=int)
    parser.add_argument("--lpips_net", type=str, default="vgg", choices=["vgg", "alex", "squeeze"], help="LPIPS backbone")
    parser.add_argument("--lpips_downsample", type=int, default=256, help="Resize H=W to this size before LPIPS (<=0 to disable)")
    
    # resume
    parser.add_argument("--resume", default=None, type=str)

    args = parser.parse_args()

    main(args)
