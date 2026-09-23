# Historical experiment code (tag `iclr2026-experiment-code`)

This commit is a **sanitized archival collection** of the implementations that were used
to run the experiments reported in

> Fanqi Wang, Weisheng Tang, Landon Harris, Hairong Qi, Dan Wilson, Igor Mezić.
> *Predictive Differential Training Guided by Training Dynamics.* ICLR 2026.

It is **not** the recommended implementation. The cleaned and maintained implementation
lives on `main` (tag `v1.0` and later). Use this snapshot only if you need to see what was
run.

The paper experiments were **not produced from a single repository state**. Two code
trees were in use, on different machines, and both changed over time. This commit
collects the relevant versions side by side. Compared with the working trees, the
following has been removed or edited (see "Sanitisation" at the end): private paths,
account names, exploratory modules and scripts that never contributed to the paper,
and internal working notes in comments. No training, prediction or masking logic was
changed.

## Layout

| Path | What it is |
|---|---|
| `kcl/`, `scripts/prediction-test/`, `setup.py` | The single-GPU research tree as of December 2025 (RTX A6000 workstation, PyTorch 2.0.0 / CUDA 11.8 / Python 3.9), reduced to the modules used for the paper. |
| `historical/local_pre_autocast_fix/` | The single-GPU PDT trainer, base trainer and argument parser as committed on 2025-11-17, i.e. **before** the mixed-precision fix of 2025-11-19 (see below). Reference copies, not importable. |
| `historical/isaac/kcl/` | The multi-GPU implementations from the ISAAC cluster tree (H100 nodes, PyTorch 2.1.2 / CUDA 12.1 container). Reference copies, not importable. |
| `historical/isaac/scripts/` | SLURM job scripts used on ISAAC. |
| `historical/isaac/container/` | Singularity definition of the ISAAC container. |

## Which implementation produced which experiment

`--trainer` names refer to `kcl/utils/hpo.py:get_trainer` of this snapshot; names in
parentheses are the ones that were used at the time when they differ.

| Implementation | File in this snapshot | Experiments |
|---|---|---|
| Single-GPU PDT, fp32 training loop | `historical/local_pre_autocast_fix/predicted_trainer_conf_post.py` (`predicted_conf`) | CIFAR-10 results of November 2024 to May 2025: Table 1 (FCN, AlexNet rows), Tables 2, 4, 5, 6, 12; Figures 2 (PDT curves), 4, 5, 6, 10(a)(b), 11, 18 |
| Single-GPU PDT, bf16 autocast (fixed) | `kcl/lib/trainers/predicted_trainer_conf_post.py` (`predicted_conf`, with `--mask_mode` and profiling options) | Rebuttal-period results of November–December 2025: Tables 7, 11, 13; Figures 16, 17, 19; and 28 of the 70 runs of Table 10 / Figures 14, 15 (those started 2025-11-22 or later) |
| Single-GPU PDT, intermediate working copy (`--mask_mode` present, fp32 training loop; not committed) | not preserved; closest versions are the two rows above | 42 of the 70 runs of Table 10 / Figures 14, 15 (started 2025-11-18) |
| Multi-GPU PDT (rank-0 DMD, broadcast) | `historical/isaac/kcl/lib/trainers/predicted_trainer_conf_post_adaptive.py` (`predicted_conf_adaptive`) | ImageNet ResNet-50 and ViT-Base runs; profiling Tables 8 and 9 |
| Multi-GPU PDT, per-layer-group DMD ("PDT-Layerwise") | `historical/isaac/kcl/lib/trainers/predicted_trainer_conf_post_adaptive_layerwise.py` (swapped in for `predicted_conf_adaptive` via the import in `hpo.py`) | ImageNet ViT-Huge PDT run only |
| Single-GPU PDT as present on ISAAC | `historical/isaac/kcl/lib/trainers/predicted_trainer_conf_post.py` | Some ImageNet ResNet-50 runs launched with `--distributed`; every rank fits the same DMD independently |
| Random acceleration baseline | `kcl/lib/trainers/randomly_accelerated.py` (`Random_accelerated`) | Figure 5 |
| Random-mask prediction baseline | `kcl/lib/trainers/predicted_trainer_conf_post_random_mask.py` (selected by swapping the commented `PredictedTrainer_conf` import in `hpo.py`, then `--trainer predicted_conf`) | Figure 6 |
| Non-selective prediction baseline | `kcl/lib/trainers/predicted_trainer.py` (selected by swapping the commented `PredictedTrainer` import in `hpo.py`, then `--trainer predicted`; this was the default mapping before commit `4495af6` of the research tree) | Figure 2 |
| Prediction/SGD switching on validation loss | `kcl/lib/trainers/predicted_trainer_switch_by_loss.py` (`predicted`, from commit `4495af6` on) | Figure 7 |
| SimSiam base and PDT trainers | `kcl/lib/trainers/simsiam_trainer.py`, `simsiam_predicted_trainer.py`, entry `scripts/prediction-test/train_test_ssl.py` | Table 3 |
| Base trainer, single GPU | `kcl/lib/trainers/trainer_base.py` | all single-GPU baselines |
| Base trainer, ISAAC (adds `--two_stage_training`) | `historical/isaac/kcl/lib/trainers/trainer_base.py` | all ImageNet baselines |
| Argument parser, ISAAC (adds `vit_huge`, SLURM/torchrun rank handling) | `historical/isaac/kcl/utils/hpo.py` | ImageNet |
| ImageNet loader, ISAAC | `historical/isaac/kcl/lib/datasets/imagenet.py` | ImageNet |
| DMD, ISAAC variants | `historical/isaac/kcl/lib/dmd/torchDMD.py`, `utils.py` | ImageNet |

Notes on behaviour that differs between versions:

- **Mixed precision.** The baseline training loop in `trainer_base.py` has used bf16
  autocast since 2023. The single-GPU PDT loop `_train_normal` trained in fp32 until
  2025-11-19; the multi-GPU trainer always used autocast. On CIFAR-10 with AlexNet the
  per-epoch time was the same either way; on ImageNet ResNet-50 the fp32 PDT loop was
  about 9 % slower per epoch than the bf16 baseline.
- **Masking.** All versions use the upper bound `(tau+1)*|delta_SGD|` and check the
  sign of the final predicted displacement only. See the README on `main` for the exact
  relation to the equations in the paper.
- **Distributed evaluation.** No cross-rank reduction of validation metrics; with a
  `DistributedSampler` on the validation set, rank 0 reports metrics on its own shard.
- **Mask files.** Every prediction epoch writes the boolean mask to
  `<log_dir>/masks/mask_epoch_XXXX.pt`.
- **Fully connected networks of Figure 2.** The 2-, 4- and 6-layer variants were
  produced by editing `kcl/lib/models/SimpleFCN.py`, which keeps the alternatives as
  commented-out class definitions.

Research-tree commits referenced above: `abf3039` (2025-11-17, last commit of the
single-GPU tree), `4495af6` (2024-09-13), `09bde8b` (2024-10-03, last commit of the ISAAC
tree; the ISAAC files in this snapshot include later uncommitted edits).

## Environments

- Workstation: Python 3.9.16, torch 2.0.0, torchvision 0.15.0, CUDA 11.8, numpy 1.23.5,
  torch_optimizer 0.3.0, datasets 4.4.1, wandb 0.19.8. Two NVIDIA RTX A6000.
- ISAAC: container from `historical/isaac/container/cuda12.def`
  (`pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime`, wandb 0.16.3). NVIDIA H100 80 GB,
  three per node.

## Sanitisation applied to this snapshot

Edits relative to the working trees. None of them changes the training, prediction or
masking logic of the trainers listed above.

- Data directories → `${DATA_DIR}`; repository root → `${PDT_ROOT}`; container image →
  `${CONTAINER_SIF}`; scratch space → `${SCRATCH}`. The ImageNet loader's hard-coded
  default directory was replaced by `os.environ.get('IMAGENET_DIR', 'data/imagenet')`.
- `--wandb_entity <user>` → `--wandb_entity ${WANDB_ENTITY}` in scripts; the parser
  default for `--wandb_entity` was changed from a user name to `None`.
- SLURM account, partition and QoS → `<account>`, `<partition>`, `<qos>`.
- Machine-specific `export CUDA_HOME/PATH/LD_LIBRARY_PATH` and `source ~/.bashrc` lines
  removed from scripts.
- Modules, classes and argument-parser branches that never contributed to a paper
  result were removed (older Koopman-acceleration trainers and optimizers, text models
  and datasets other than the AG News LSTM, alternative DMD implementations, plotting
  helpers). Unused imports of the removed modules were deleted from the remaining
  trainers. Trainer selection, command-line options and defaults of the remaining
  experiments were left as they were.
- Internal working notes in comments and docstrings were removed or replaced by neutral
  technical descriptions; non-English comments were translated; decorative non-ASCII
  characters in log messages were removed.
- `setup.py` no longer lists an internal DMD package and the DALI wheel as
  dependencies; the DMD implementation used by PDT is in `kcl/lib/dmd`.
- A small byte-formatting helper (`pretty_size`) that had been taken from a Stack Overflow
  answer was re-implemented independently.
- Third-party attributions were collected in `THIRD_PARTY_NOTICES.md`.

## Not included

Jupyter notebooks, wandb exports, result CSV files, figure files, training logs, mask
files, checkpoints, rebuttal notes, and exploratory scripts and modules that were not
used for the paper.
