import os
import re
import json
import argparse
import random
from glob import glob


FILENAME_REGEX = re.compile(
    r"^(?P<prefix>route)_(?P<pt>pt\d{3})_h(?P<h>\d{3})_p(?P<p>\d{2})_fov(?P<fov>\d{2,3})\.(?P<ext>jpg|png)$",
    re.IGNORECASE,
)


def parse_name(filename: str):
    name = os.path.basename(filename)
    m = FILENAME_REGEX.match(name)
    if not m:
        return None
    gd = m.groupdict()
    return {
        "prefix": gd["prefix"],
        "pt": gd["pt"],
        "h": int(gd["h"]),
        "p": int(gd["p"]),
        "fov": int(gd["fov"]),
        "ext": gd["ext"].lower(),
    }


def build_index(label_dir: str):
    index = {}
    # support both jpg and png in label dir
    for path in glob(os.path.join(label_dir, "route_*.*")):
        meta = parse_name(path)
        if not meta:
            continue
        key = f"{meta['prefix']}_{meta['pt']}_h{meta['h']:03d}_p{meta['p']:02d}_fov{meta['fov']}"
        index[key] = path
    return index


def find_ghost(ghost_dir: str, meta_key: str):
    # try both png/jpg for ghosts
    png = os.path.join(ghost_dir, f"{meta_key}.png")
    if os.path.exists(png):
        return png
    jpg = os.path.join(ghost_dir, f"{meta_key}.jpg")
    if os.path.exists(jpg):
        return jpg
    return None


def neighbor_key(meta_key: str, delta_h: int):
    # meta_key example: route_pt002_h180_p00_fov90
    try:
        prefix, pt, hpart, ppart, fovpart = meta_key.split("_")
        h = int(hpart[1:])
        new_h = (h + delta_h) % 360
        return f"{prefix}_{pt}_h{new_h:03d}_{ppart}_{fovpart}"
    except Exception:
        return None


def split_by_pt(keys, test_ratio: float, seed: int):
    # keep scenes (pt) separated between splits
    pts = {}
    for k in keys:
        pt = k.split("_")[1]  # ptXXX
        pts.setdefault(pt, []).append(k)
    pt_ids = sorted(list(pts.keys()))
    rnd = random.Random(seed)
    rnd.shuffle(pt_ids)
    n_test = max(1, int(len(pt_ids) * test_ratio))
    test_pts = set(pt_ids[:n_test])
    train_keys, test_keys = [], []
    for pt, group in pts.items():
        if pt in test_pts:
            test_keys.extend(group)
        else:
            train_keys.extend(group)
    return train_keys, test_keys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ghost_dir", type=str, default="/data2/peilincai/Difix3D/datasets/ghosts")
    parser.add_argument("--label_dir", type=str, default="/data2/peilincai/Difix3D/datasets/label")
    parser.add_argument("--output_json", type=str, default="/data2/peilincai/Difix3D/datasets/difix3d.json")
    parser.add_argument("--prompt", type=str, default="remove degradation")
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--both_refs", action="store_true", help="Create two entries per view with left and right refs.")
    parser.add_argument("--single_ref_side", type=str, default="left", choices=["left", "right"], help="If not both_refs, which neighbor to use.")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)

    label_index = build_index(args.label_dir)
    keys = sorted(label_index.keys())

    train_keys, test_keys = split_by_pt(keys, test_ratio=args.test_ratio, seed=args.seed)

    def build_split(keys_split):
        split = {}
        for base_key in keys_split:
            target_path = label_index.get(base_key)
            ghost_path = find_ghost(args.ghost_dir, base_key)
            if ghost_path is None:
                continue

            # neighbors from label as reference
            left_key = neighbor_key(base_key, +45)
            right_key = neighbor_key(base_key, -45)
            left_path = label_index.get(left_key) if left_key else None
            right_path = label_index.get(right_key) if right_key else None

            entries = []
            if args.both_refs:
                if left_path:
                    entries.append(("left", left_path))
                if right_path:
                    entries.append(("right", right_path))
            else:
                if args.single_ref_side == "left" and left_path:
                    entries.append(("left", left_path))
                elif args.single_ref_side == "right" and right_path:
                    entries.append(("right", right_path))

            if not entries:
                # fallback to no ref if neighbors missing
                data_id = base_key
                split[data_id] = {
                    "image": ghost_path,
                    "target_image": target_path,
                    "prompt": args.prompt,
                }
                continue

            for side, ref_path in entries:
                data_id = f"{base_key}_{side}"
                split[data_id] = {
                    "image": ghost_path,
                    "target_image": target_path,
                    "ref_image": ref_path,
                    "prompt": args.prompt,
                }
        return split

    dataset = {
        "train": build_split(train_keys),
        "test": build_split(test_keys),
    }

    with open(args.output_json, "w") as f:
        json.dump(dataset, f, indent=2)

    num_train = len(dataset["train"]) 
    num_test = len(dataset["test"]) 
    print(f"Wrote {num_train} train and {num_test} test entries to {args.output_json}")


if __name__ == "__main__":
    main()


