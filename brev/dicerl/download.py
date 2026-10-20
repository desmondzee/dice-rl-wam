import os, sys
from huggingface_hub import snapshot_download
root = os.path.expanduser("~/Documents/dice/dicerl")
for env in sys.argv[1:]:
    for repo, dest in (("wintermelontree/robomimic-pretrain-data", f"{root}/data_dir/robomimic/{env}-low-dim/ph_pretrain"),
                       ("wintermelontree/robomimic-finetune-data", f"{root}/data_dir/robomimic/{env}-low-dim/ph_finetune")):
        p = snapshot_download(repo, repo_type="dataset", allow_patterns=[f"{env}_low_dim/*"], local_dir=f"{root}/hf/{repo.split('/')[1]}")
        os.makedirs(dest, exist_ok=True)
        for f in os.listdir(f"{p}/{env}_low_dim"):
            os.replace(f"{p}/{env}_low_dim/{f}", f"{dest}/{f}")
    for repo, dest in (("wintermelontree/robomimic-pretrain-checkpoints", f"{root}/log_dir/robomimic-pretrain/pretrained_bc_policy_{env}_low_dim"),
                       ("wintermelontree/robomimic-finetune-checkpoints", f"{root}/log_dir/robomimic-finetune/finetune_rl_policy_{env}_low_dim")):
        p = snapshot_download(repo, repo_type="model", allow_patterns=[f"{env}_low_dim/**"], local_dir=f"{root}/hf/{repo.split('/')[1]}")
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        if not os.path.exists(dest):
            os.replace(f"{p}/{env}_low_dim", dest)
    print("done", env, flush=True)
