import argparse
import os
import re
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Tuple as Tup

from diffusers.utils import load_image
from PIL import Image

from pipeline_difix import DifixPipeline


FILENAME_REGEX = re.compile(
    r"^(?P<route>[^_]+)_pt(?P<pt>\d+)_h(?P<h>\d+)_p(?P<p>\d+)_fov(?P<fov>\d+)\.(?P<ext>png|jpg|jpeg)$",
    re.IGNORECASE,
)


def parse_metadata(filename: str) -> Optional[dict]:
    match = FILENAME_REGEX.match(filename)
    if not match:
        return None
    groups = match.groupdict()
    return {
        "route": groups["route"],
        "pt": int(groups["pt"]),
        "h": int(groups["h"]) % 360,
        "p": int(groups["p"]),
        "fov": int(groups["fov"]),
        "ext": groups["ext"].lower(),
    }


def build_basename(route: str, pt: int, h: int, p: int, fov: int, ext: str) -> str:
    return f"{route}_pt{pt:03d}_h{h:03d}_p{p:02d}_fov{fov}.{ext}"


def find_image_with_meta_index(
    images_index: Dict[Tup[str, int, int, int, int], Path],
    route: str,
    pt: int,
    h: int,
    p: int,
    fov: int,
) -> Optional[Path]:
    return images_index.get((route, pt, h % 360, p, fov))


def find_reference_image_index(
    images_index: Dict[Tup[str, int, int, int, int], Path], meta: dict
) -> Optional[Path]:
    # Prefer +45°, then -45°. Fall back to any file with |delta| == 45° if present.
    route, pt, h, p, fov = meta["route"], meta["pt"], meta["h"], meta["p"], meta["fov"]
    plus_h = (h + 45) % 360
    minus_h = (h - 45) % 360

    # Direct tries
    for target_h in (plus_h, minus_h):
        path = images_index.get((route, pt, target_h, p, fov))
        if path is not None:
            return path

    # Fallback: scan keys for any file with |delta| == 45° that matches route/pt/p/fov
    for (r, pt_k, h_k, p_k, fov_k), entry in images_index.items():
        if r == route and pt_k == pt and p_k == p and fov_k == fov:
            delta = (h_k - h) % 360
            if delta == 45 or delta == 315:
                return entry
    return None


def ensure_same_size(left: Image.Image, right: Image.Image) -> Tuple[Image.Image, Image.Image]:
    # Resize right to match left size if they differ
    if left.size == right.size:
        return left, right
    return left, right.resize(left.size, Image.BICUBIC)


def compose_side_by_side(left: Image.Image, right: Image.Image) -> Image.Image:
    left, right = ensure_same_size(left, right)
    w1, h1 = left.size
    w2, h2 = right.size
    out = Image.new("RGB", (w1 + w2, max(h1, h2)))
    out.paste(left, (0, 0))
    out.paste(right, (w1, 0))
    return out


def list_image_files(directory: Path) -> List[Path]:
    return sorted(
        [
            p
            for p in directory.iterdir()
            if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        ]
    )


def index_images(images_dir: Path) -> Dict[Tup[str, int, int, int, int], Path]:
    images_index: Dict[Tup[str, int, int, int, int], Path] = {}
    if not images_dir.exists():
        return images_index
    for p in images_dir.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        meta = parse_metadata(p.name)
        if meta is None:
            continue
        key = (meta["route"], meta["pt"], meta["h"], meta["p"], meta["fov"])
        # Do not override existing entries; keep first found
        images_index.setdefault(key, p)
    return images_index


def main():
    parser = argparse.ArgumentParser(description="Batch Difix on ghosts with 45° ref from images")
    parser.add_argument(
        "--ghosts_dir",
        type=str,
        default="/data2/peilincai/Difix3D/assets/ghosts",
        help="Directory containing ghost images",
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        required=True,
        help="Directory containing reference/original images",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run the pipeline on",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="remove degradation",
        help="Prompt for DifixPipeline",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=1,
        help="Number of inference steps",
    )
    parser.add_argument(
        "--timesteps",
        type=str,
        default="199",
        help="Comma-separated timesteps (e.g., '199' or '199,150')",
    )
    args = parser.parse_args()

    ghosts_dir = Path(args.ghosts_dir)
    images_dir = Path(args.images_dir)
    outputs_dir = ghosts_dir.parent / "outputs"
    constra_dir = ghosts_dir.parent / "constra"

    outputs_dir.mkdir(parents=True, exist_ok=True)
    constra_dir.mkdir(parents=True, exist_ok=True)

    # Load pipeline once
    pipe = DifixPipeline.from_pretrained("nvidia/difix_ref", trust_remote_code=True)
    pipe.to(args.device)

    timestep_list = [int(t.strip()) for t in args.timesteps.split(",") if t.strip()]

    # Build images index recursively
    images_index = index_images(images_dir)
    if not images_index:
        print(f"No reference/original images found under: {images_dir} (recursive)")
        return

    ghost_files = list_image_files(ghosts_dir)
    if not ghost_files:
        print(f"No images found in ghosts_dir: {ghosts_dir}")
        return

    for ghost_path in ghost_files:
        meta = parse_metadata(ghost_path.name)
        if meta is None:
            print(f"Skip unrecognized filename format: {ghost_path.name}")
            continue

        print(f"Processing {ghost_path.name} ...")

        # Find reference image (+/- 45°)
        ref_path = find_reference_image_index(images_index, meta)
        if ref_path is None:
            print(f"  Could not find reference image at ±45° for {ghost_path.name}; skipping.")
            continue

        # Load inputs
        try:
            input_image = load_image(str(ghost_path))
            ref_image = load_image(str(ref_path))
        except Exception as e:
            print(f"  Failed to load images: {e}; skipping.")
            continue

        # Run pipeline
        try:
            result = pipe(
                args.prompt,
                image=input_image,
                ref_image=ref_image,
                num_inference_steps=args.num_inference_steps,
                timesteps=timestep_list,
                guidance_scale=0.0,
            )
            output_image = result.images[0]
        except Exception as e:
            print(f"  Pipeline failed: {e}; skipping.")
            continue

        # Save output
        out_name = ghost_path.stem + ".png"
        out_path = outputs_dir / out_name
        try:
            output_image.save(out_path)
            print(f"  Saved output -> {out_path}")
        except Exception as e:
            print(f"  Failed to save output: {e}")

        # Build side-by-side with original image (same view) from images_dir
        orig_path = find_image_with_meta_index(
            images_index, meta["route"], meta["pt"], meta["h"], meta["p"], meta["fov"]
        )
        if orig_path is None:
            print(f"  Original image not found in images_dir for {ghost_path.name}; skip constra.")
            continue

        try:
            original_image = load_image(str(orig_path))
            constra_image = compose_side_by_side(original_image, output_image)
            constra_path = constra_dir / out_name
            constra_image.save(constra_path)
            print(f"  Saved comparison -> {constra_path}")
        except Exception as e:
            print(f"  Failed to create/save comparison: {e}")


if __name__ == "__main__":
    main()


