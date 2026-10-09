import argparse
import hashlib
import os

import modal

CHUNK = 64 << 20
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--volume", default="dice-lingbot-rl-runs")
VOLUME = _parser.parse_known_args()[0].volume
app = modal.App("modal-volume-tools")
vol = modal.Volume.from_name(VOLUME)


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


@app.function(volumes={"/v": vol}, timeout=3600, cpu=2, memory=2048)
def sha_dir(prefix):
    vol.reload()
    root = os.path.join("/v", prefix)
    paths = [root] if os.path.isfile(root) else [os.path.join(d, n) for d, _, fs in os.walk(root) for n in fs]
    return sorted((os.path.relpath(p, "/v"), os.path.getsize(p), _sha(p)) for p in paths)


@app.function(volumes={"/v": vol}, timeout=600, cpu=1, memory=1024)
def info(path):
    vol.reload()
    full = os.path.join("/v", path)
    return os.path.getsize(full), _sha(full)


@app.function(volumes={"/v": vol}, timeout=3600, cpu=2, memory=4096)
def fetch(path, index):
    vol.reload()
    with open(os.path.join("/v", path), "rb") as f:
        f.seek(index * CHUNK)
        data = f.read(CHUNK)
    return hashlib.sha256(data).hexdigest(), data


@app.local_entrypoint()
def main(cmd: str, prefix: str = "", path: str = "", out: str = "", volume: str = VOLUME):
    if cmd == "sha":
        for rel, size, digest in sha_dir.remote(prefix):
            print(f"{digest}  {size}  {rel}")
    elif cmd == "fetch":
        size, digest = info.remote(path)
        h = hashlib.sha256()
        with open(out, "wb") as f:
            for i in range((size + CHUNK - 1) // CHUNK):
                for _ in range(5):
                    chunk_sha, data = fetch.remote(path, i)
                    if hashlib.sha256(data).hexdigest() == chunk_sha:
                        break
                else:
                    raise SystemExit(f"chunk {i} failed")
                f.write(data)
                h.update(data)
        local = h.hexdigest()
        print(f"{path} size={size} remote_sha={digest} local_sha={local}")
        if local != digest:
            raise SystemExit("whole-file sha mismatch")
    else:
        raise SystemExit("cmd must be sha or fetch")
