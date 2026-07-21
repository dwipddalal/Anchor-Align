# Conda environment snapshots

Frozen environment exports captured from the Delta GH200 host on which experiments were run. Use these on another server to reproduce the exact dependency set.

## Which env does what

| Env | Used for |
|---|---|
| `vla-adapter-pt27` | CALVIN batched eval (`vla-scripts/evaluate_calvin_batched.py`), CALVIN slurm scripts |
| `vla-adapter` | LIBERO eval (spatial / object / goal / 10 / pro / plus) |
| `calvin` | CALVIN simulator client (`calvin_env`, hydra configs, bullet-cpp) |

The slurm scripts read `CONDA_ENV` with sane defaults, e.g.:
```
CONDA_ENV="${CONDA_ENV:-vla-adapter-pt27}"
```

## Recreate an env

Two formats per env are provided:

1. **Conda YAML** (`<env>.yml`) — full conda + pip spec with versions, no build hashes (portable across hosts). Recreate with:
   ```bash
   conda env create -f setup/envs/vla-adapter-pt27.yml
   ```

2. **Pip requirements** (`<env>-requirements.txt`) — just the pip packages. Use this if you'd rather start from a fresh conda env you control:
   ```bash
   conda create -n vla-adapter-pt27 python=3.10
   conda activate vla-adapter-pt27
   pip install -r setup/envs/vla-adapter-pt27-requirements.txt
   ```

## Notes / gotchas

- **GH200 / aarch64**: the YAML pins are aarch64 builds. On x86 hosts use the pip requirements path or remove the arch-specific conda pins.
- **PyTorch / CUDA**: `vla-adapter-pt27` is PyTorch 2.7 with Triton 3.3 on cuda-compat 12.8. The companion script `setup/setup_pt27_env.sh` documents the original install path.
- **CALVIN dependencies**: clone the `CALVIN/` submodule separately (`git clone https://github.com/mees/calvin.git`) and `pip install -e calvin/calvin_env calvin/calvin_models` inside the `calvin` env after creating from YAML.
