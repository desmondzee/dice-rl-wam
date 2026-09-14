import argparse
import getpass


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--service", choices=("wandb", "hf"), default="wandb")
    args = parser.parse_args(argv)
    import modal

    name, variable, label = (
        ("dice-lingbot-wandb", "WANDB_API_KEY", "W&B API key")
        if args.service == "wandb"
        else ("dice-lingbot-hf", "HF_TOKEN", "Hugging Face read token")
    )
    key = getpass.getpass(f"{label} for Modal (hidden): ").strip()
    if not key or "\n" in key or "\r" in key:
        raise ValueError("A nonempty single-line token is required")
    modal.Secret.objects.create(name, {variable: key})
    print(f"Created Modal secret {name} ({variable}).")


if __name__ == "__main__":
    main()
