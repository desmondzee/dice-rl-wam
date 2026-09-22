import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

VOLUME = "dice-lingbot-rl-runs"


def volume_command(profile, *args, **kwargs):
    env = {**os.environ, "MODAL_PROFILE": profile}
    return subprocess.run([sys.executable, "-m", "modal", "volume", *args], env=env, **kwargs)


def is_present(profile, volume, remote):
    listing = volume_command(profile, "ls", volume, str(Path(remote).parent), capture_output=True, text=True)
    if listing.returncode != 0:
        return False
    return any(Path(line.strip()).name == Path(remote).name for line in listing.stdout.splitlines() if line.strip())


def copy_checkpoint(remote, source_profile, dest_profile, volume=VOLUME):
    if is_present(dest_profile, volume, remote):
        return "present"
    volume_command(dest_profile, "create", volume, capture_output=True)
    with tempfile.TemporaryDirectory() as temporary:
        volume_command(source_profile, "get", volume, remote, temporary, check=True)
        local = Path(temporary) / Path(remote).name
        if not local.is_file():
            raise FileNotFoundError(f"{remote} was not downloaded from {source_profile}")
        volume_command(dest_profile, "put", volume, str(local), remote, check=True)
    return "copied"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--residual", required=True)
    parser.add_argument("--source-profile", default="desmond-zee")
    parser.add_argument("--dest-profile", default="nobel")
    parser.add_argument("--volume", default=VOLUME)
    args = parser.parse_args()
    print(f"{copy_checkpoint(args.residual, args.source_profile, args.dest_profile, args.volume)}: {args.residual}")


if __name__ == "__main__":
    main()
