import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from script.lingbot_sft_config import UPSTREAM_REVISION


_REPLACEMENTS = {
    Path("wan_va/train.py"): (
        "from dataset import MultiLatentLeRobotDataset",
        "from script.lingbot_sft_data import LatentDataset as MultiLatentLeRobotDataset",
    ),
    Path("wan_va/modules/model.py"): (
        """try:
    from flash_attn_interface import flash_attn_func
except:
    from flash_attn import flash_attn_func""",
        """try:
    from flash_attn_interface import flash_attn_func
except ImportError:
    try:
        from flash_attn import flash_attn_func
    except ImportError:
        def flash_attn_func(*args, **kwargs):
            raise RuntimeError("FlashAttention is unavailable; this SFT adapter requires attn_mode='flex'")""",
    ),
}


def _revision(root):
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _replace_once(path, old, new):
    text = path.read_text()
    old_count = text.count(old)
    new_count = text.count(new)
    if old_count == 0 and new_count == 1:
        return False
    if old_count != 1 or new_count != 0:
        raise RuntimeError(f"Unexpected replacement count in {path}: old={old_count}, new={new_count}")
    path.write_text(text.replace(old, new, 1))
    return True


def apply_patch(root: Path):
    root = Path(root)
    revision = _revision(root)
    if revision != UPSTREAM_REVISION:
        raise RuntimeError(f"Upstream revision mismatch: expected {UPSTREAM_REVISION}, got {revision}")
    changed = {}
    for relative, (old, new) in _REPLACEMENTS.items():
        path = root / relative
        if _replace_once(path, old, new):
            changed[str(relative)] = True
        else:
            changed[str(relative)] = False
    hashes = {
        str(relative): hashlib.sha256((root / relative).read_bytes()).hexdigest()
        for relative in _REPLACEMENTS
    }
    return {"revision": revision, "changed": changed, "hashes": hashes}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(apply_patch(args.root), sort_keys=True))


if __name__ == "__main__":
    main()
