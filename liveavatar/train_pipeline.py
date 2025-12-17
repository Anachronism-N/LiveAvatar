import argparse
import json
import os
import random
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchaudio
from decord import VideoReader
from PIL import Image
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision import transforms

from liveavatar.models.wan.causal_s2v_pipeline import WanS2V
from liveavatar.models.wan.wan_2_2.configs.wan_s2v_14B_modified import s2v_14B
from liveavatar.utils.args_config import parse_args_for_training_config


@dataclass
class SampleRecord:
    """Simple container for each training sample."""

    video: str
    audio: str
    prompt: str
    ref_image: Optional[str] = None


class AvatarDataset(Dataset):
    """
    Minimal dataset for LiveAvatar IT2V training.

    Expected meta files: JSON/JSONL with keys
        - video: path to a video file or directory of frames
        - audio: path to the aligned audio file
        - prompt: text prompt
        - ref_image (optional): override reference image (otherwise first frame is used)
    """

    def __init__(
        self,
        meta_files: List[str],
        num_frames: int,
        max_wh: int,
        frame_interval: int,
        sample_fps: int,
        device: torch.device,
        use_audio: bool,
    ):
        self.num_frames = num_frames
        self.max_wh = max_wh
        self.frame_interval = frame_interval
        self.sample_fps = sample_fps
        self.device = device
        self.use_audio = use_audio
        self.samples: List[SampleRecord] = []
        for meta_path in meta_files:
            if not os.path.exists(meta_path):
                raise FileNotFoundError(f"Meta file not found: {meta_path}")
            with open(meta_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    meta = json.loads(line)
                    self.samples.append(
                        SampleRecord(
                            video=meta["video"],
                            audio=meta.get("audio"),
                            prompt=meta.get("prompt", ""),
                            ref_image=meta.get("ref_image"),
                        )
                    )

        self.tensor_transform = transforms.ToTensor()

    def __len__(self):
        return len(self.samples)

    def _load_frames(self, path: str) -> np.ndarray:
        vr = VideoReader(path)
        total = len(vr)
        if total <= 0:
            raise ValueError(f"Empty video: {path}")
        # Uniform sampling with interval control
        indices = np.linspace(
            0, max(total - 1, 0), self.num_frames * self.frame_interval, dtype=int
        )[:: self.frame_interval]
        indices = np.clip(indices, 0, total - 1)
        frames = vr.get_batch(indices).asnumpy()  # [T, H, W, 3] uint8
        return frames

    def _resize_and_pad(self, image: Image.Image) -> Image.Image:
        """Resize keeping aspect ratio; pad to multiples of 64."""
        w, h = image.size
        scale = min(self.max_wh / float(h), self.max_wh / float(w), 1.0)
        nh, nw = int(h * scale), int(w * scale)
        image = image.resize((nw, nh), Image.BICUBIC)

        pad_h = (64 - nh % 64) % 64
        pad_w = (64 - nw % 64) % 64
        pad_top = pad_h // 2
        pad_left = pad_w // 2
        new_img = Image.new("RGB", (nw + pad_w, nh + pad_h), (0, 0, 0))
        new_img.paste(image, (pad_left, pad_top))
        return new_img

    def _process_frames(self, frames: np.ndarray) -> torch.Tensor:
        processed = []
        for frame in frames:
            img = Image.fromarray(frame.astype(np.uint8))
            img = self._resize_and_pad(img)
            tensor = self.tensor_transform(img) * 2 - 1.0  # [-1, 1]
            processed.append(tensor)
        video = torch.stack(processed, dim=1)  # [C, T, H, W]
        return video

    def _load_audio(self, path: str, target_len: float) -> torch.Tensor:
        if path is None:
            raise ValueError("Audio path is None but use_audio=True.")

        wav, sr = torchaudio.load(path)
        wav = wav.mean(dim=0, keepdim=True)  # mono
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        target_samples = int(target_len * 16000)
        if wav.shape[-1] < target_samples:
            pad = target_samples - wav.shape[-1]
            wav = F.pad(wav, (0, pad))
        else:
            wav = wav[..., :target_samples]
        return wav

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        record = self.samples[idx]
        frames = self._load_frames(record.video)
        video_tensor = self._process_frames(frames)
        ref_img = (
            Image.open(record.ref_image).convert("RGB")
            if record.ref_image is not None
            else Image.fromarray(frames[0].astype(np.uint8))
        )
        ref_img = self.tensor_transform(self._resize_and_pad(ref_img)) * 2 - 1.0
        audio_seconds = self.num_frames / float(self.sample_fps)
        if self.use_audio:
            if record.audio is None:
                raise ValueError("Audio is required but missing in metadata.")
            audio_tensor = self._load_audio(record.audio, target_len=audio_seconds)
        else:
            # create a silent audio tensor for video-only training or ablation
            audio_tensor = torch.zeros(1, int(audio_seconds * 16000))
        return {
            "video": video_tensor,
            "ref": ref_img,
            "audio": audio_tensor,
            "prompt": record.prompt,
        }


def init_distributed() -> Tuple[int, int, int]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    else:
        rank, world_size, local_rank = 0, 1, 0
    return rank, world_size, local_rank


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_optimizer(
    model: WanS2V, training_cfg: dict
) -> torch.optim.Optimizer:
    lr = float(training_cfg.get("learning_rate", 5e-5))
    audio_lr = float(training_cfg.get("audio_learning_rate", lr))
    param_groups = []

    # Freeze everything first
    for p in model.noise_model.parameters():
        p.requires_grad_(False)
    for p in model.text_encoder.model.parameters():
        p.requires_grad_(False)
    for p in model.vae.model.parameters():
        p.requires_grad_(False)
    for p in model.audio_encoder.model.parameters():
        p.requires_grad_(False)

    # Enable LoRA params
    lora_params = [
        p for n, p in model.noise_model.named_parameters() if "lora_" in n
    ]
    for p in lora_params:
        p.requires_grad_(True)
    if lora_params:
        param_groups.append({"params": lora_params, "lr": lr})

    # Optional audio finetuning
    if training_cfg.get("use_audio", True):
        audio_params = [
            p
            for p in model.audio_encoder.model.parameters()
            if p.requires_grad is False
        ]
        for p in audio_params:
            p.requires_grad_(True)
        if audio_params:
            param_groups.append({"params": audio_params, "lr": audio_lr})

    if not param_groups:
        raise RuntimeError("No trainable parameters were found for optimizer.")

    return torch.optim.AdamW(param_groups, weight_decay=0.01)


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, List[torch.Tensor]]:
    return {
        "video": [item["video"] for item in batch],
        "ref": [item["ref"] for item in batch],
        "audio": [item["audio"] for item in batch],
        "prompt": [item["prompt"] for item in batch],
    }


def prepare_latents(model: WanS2V, batch: Dict[str, List[torch.Tensor]], training_cfg: dict, device: torch.device):
    videos = [b.to(device, dtype=model.param_dtype) for b in batch["video"]]
    refs = [b.to(device, dtype=model.param_dtype) for b in batch["ref"]]
    audios = [b.to(device) for b in batch["audio"]]

    # Encode video & ref
    clean_latents = torch.stack(model.vae.encode(videos)).to(model.param_dtype)
    # temporal compression factor is 4
    lat_len = clean_latents.shape[2]

    base_ref = torch.stack(refs)  # [B, C, H, W]
    ref_pixel_values = base_ref.unsqueeze(2).repeat(1, 1, 5, 1, 1)
    ref_latents = torch.stack(
        model.vae.encode([rp for rp in ref_pixel_values])
    )[:, :, 1:].to(model.param_dtype)

    motion_frames = training_cfg.get("motion_frames_override", model.motion_frames)
    motion_pixels = base_ref.unsqueeze(2).repeat(1, 1, motion_frames, 1, 1)
    motion_latents = torch.stack(
        model.vae.encode([mp for mp in motion_pixels])
    ).to(model.param_dtype)

    cond_states = torch.zeros_like(clean_latents)

    return clean_latents, ref_latents, motion_latents, cond_states, lat_len, audios


def train_one_step(
    model: WanS2V,
    optimizer: torch.optim.Optimizer,
    scheduler,
    batch: Dict[str, List[torch.Tensor]],
    training_cfg: dict,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.noise_model.train()
    model.text_encoder.model.eval()
    model.vae.model.eval()
    if training_cfg.get("use_audio", True):
        model.audio_encoder.model.train()
    else:
        model.audio_encoder.model.eval()

    clean_latents, ref_latents, motion_latents, cond_states, lat_len, audios = prepare_latents(
        model, batch, training_cfg, device
    )

    # timestep
    t_scalar = torch.randint(
        0, scheduler.num_train_timesteps, (1,), device=device
    ).item()
    t_tensor = torch.full(
        (clean_latents.shape[0], lat_len),
        float(t_scalar),
        device=device,
        dtype=model.param_dtype,
    )

    noise = torch.randn_like(clean_latents)
    sigma_idx = torch.argmin((scheduler.timesteps - torch.tensor(t_scalar)).abs())
    sigma = scheduler.sigmas[sigma_idx].to(clean_latents.device, dtype=clean_latents.dtype)
    noisy_latents = (1 - sigma) * clean_latents + sigma * noise

    with torch.cuda.amp.autocast(dtype=model.param_dtype):
        audio_embs = []
        drop_audio_p = float(training_cfg.get("drop_audio", 0.0))
        for audio in audios:
            audio_emb, _ = model.encode_audio_training(
                audio, infer_frames=training_cfg["num_frames"], fps=model.fps
            )
            if (not training_cfg.get("use_audio", True)) or (
                drop_audio_p > 0 and random.random() < drop_audio_p
            ):
                audio_emb = torch.zeros_like(audio_emb)
            audio_embs.append(audio_emb)
        audio_emb = torch.cat(audio_embs, dim=0)

        context, _ = model.encode_prompt(batch["prompt"], offload_model=False)
        lat_motion_frames = (model.motion_frames + 3) // 4

        noise_pred = model.noise_model(
            [noisy_latents[i] for i in range(noisy_latents.shape[0])],
            t=t_tensor,
            context=context,
            seq_len=None,
            ref_latents=ref_latents,
            motion_latents=motion_latents,
            cond_states=cond_states,
            audio_input=audio_emb,
            motion_frames=[model.motion_frames, lat_motion_frames],
            drop_motion_frames=training_cfg.get("drop_motion_frames", 0) > 0,
            drop_part_motion_frames=training_cfg.get("drop_part_motion_frames", 0)
            > 0,
        )
        noise_pred = torch.cat(noise_pred, dim=0)
        target = noise - noisy_latents
        weight = scheduler.training_weight(
            torch.tensor(t_scalar, device=scheduler.timesteps.device)
        )
        loss = F.mse_loss(noise_pred, target, reduction="none").mean()
        loss = loss * weight.to(loss.device)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.detach(), torch.tensor(t_scalar, device=device)


def save_checkpoint(model: WanS2V, optimizer, step: int, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    state = {
        "noise_model": model.noise_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
    }
    torch.save(state, os.path.join(save_dir, f"step_{step}.pt"))


def parse_train_args():
    parser = argparse.ArgumentParser(description="LiveAvatar training pipeline")
    parser.add_argument("--config", required=True, help="Path to YAML training config")
    parser.add_argument(
        "--save_dir", default="checkpoints/liveavatar", help="Where to save checkpoints"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--max_steps", type=int, default=None)
    return parser.parse_args()


def main():
    rank, world_size, local_rank = init_distributed()
    args = parse_train_args()
    seed_everything(args.seed + rank)

    training_cfg = parse_args_for_training_config(args.config)
    if args.max_steps is not None:
        training_cfg["max_steps"] = args.max_steps
    training_cfg.setdefault("num_frames", 48)
    if not training_cfg.get("dataset_path"):
        raise ValueError("dataset_path is empty; please provide meta files in the YAML config.")
    if "dataset" not in training_cfg:
        raise ValueError("dataset section missing in config; expected keys like max_wh and frame_interval.")

    model_cfg = deepcopy(s2v_14B)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    model = WanS2V(
        config=model_cfg,
        checkpoint_dir=training_cfg["ckpt_dir"],
        device_id=device.index if device.type == "cuda" else 0,
        rank=rank,
        is_training=True,
        convert_model_dtype=training_cfg.get("convert_model_dtype", True),
        drop_part_motion_frames=training_cfg.get("drop_part_motion_frames", 0) > 0,
        t5_cpu=training_cfg.get("t5_cpu", True),
    )

    if training_cfg.get("train_architecture", "lora") == "lora":
        model.add_lora_to_model(
            model.noise_model,
            lora_rank=training_cfg.get("lora_rank", 128),
            lora_alpha=training_cfg.get("lora_alpha", 64),
            lora_target_modules=training_cfg.get(
                "lora_target_modules", "q,k,v,o,ffn.0,ffn.2"
            ),
            init_lora_weights=training_cfg.get("init_lora_weights", "kaiming"),
            pretrained_lora_path=training_cfg.get("pretrained_lora_path"),
        )

    if training_cfg.get("use_gradient_checkpointing", False):
        model.noise_model.enable_gradient_checkpointing()

    optimizer = build_optimizer(model, training_cfg)
    scheduler = model.scheduler
    scheduler.set_timesteps(training_cfg.get("infer_steps", 20), training=True)

    dataset = AvatarDataset(
        meta_files=training_cfg.get("dataset_path", []),
        num_frames=training_cfg["num_frames"],
        max_wh=training_cfg["dataset"]["max_wh"],
        frame_interval=training_cfg["dataset"].get("frame_interval", 1),
        sample_fps=model.fps,
        device=device,
        use_audio=training_cfg.get("use_audio", True),
    )

    sampler = (
        DistributedSampler(dataset, shuffle=True)
        if world_size > 1
        else None
    )
    dataloader = DataLoader(
        dataset,
        batch_size=training_cfg.get("deepspeed_config", {}).get(
            "train_micro_batch_size_per_gpu", 1
        ),
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=training_cfg.get("dataloader_num_workers", 0),
        collate_fn=collate_fn,
        drop_last=True,
    )

    global_step = 0
    for epoch in range(10_000):  # large upper bound; stop via max_steps
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in dataloader:
            loss, timestep = train_one_step(
                model, optimizer, scheduler, batch, training_cfg, device
            )
            if global_step % args.log_interval == 0 and rank == 0:
                print(
                    f"[Step {global_step}] loss={loss.item():.4f} timestep={timestep.item()}"
                )
            global_step += 1
            if global_step % args.save_interval == 0 and rank == 0:
                save_checkpoint(model, optimizer, global_step, args.save_dir)
            if global_step >= training_cfg.get("max_steps", 100000):
                break
        if global_step >= training_cfg.get("max_steps", 100000):
            break

    if rank == 0:
        save_checkpoint(model, optimizer, global_step, args.save_dir)


if __name__ == "__main__":
    main()
