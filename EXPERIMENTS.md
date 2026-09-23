# Experiment configurations

This file maps the experiments of the paper to the scripts in this repository. The
scripts are **configurations**: they set up the setting of each experiment (model,
dataset, optimizer, schedule, PDT hyper-parameters) with the cleaned implementations of
this release; the implementations that produced the reported results are preserved under
the git tag `iclr2026-experiment-code` (see README, "Implementation notes"). Raw training
logs and the plotting code used for the figures are not part of this release; metrics of
every run are written to `<log_dir>/metrics.csv` (see README, "Outputs").

Unless noted otherwise, PDT uses the default hyper-parameters of the paper:
`tau = 5`, `T_i = 1`, `T_0 = 5`, `h = 5` (`--predicted_num 5 --predict_epoch_interval 1
--predict_start_epoch 5 --n_past_weights 5`). Every script accepts `SEED=<int>` and
extra arguments that are passed through to `scripts/train.py`;
`bash scripts/run_seeds.sh <script> 0 100 200 300 400` repeats a script over seeds.

## CIFAR-10 (single GPU)

| Paper | Script | Notes |
|---|---|---|
| Fig. 2: FC networks, SGD vs non-selective prediction vs PDT | `FC_LAYERS={2,4,6} scripts/cifar10/fc_networks.sh` | lr 0.01, 30 epochs, predictions every 3 epochs |
| Fig. 4, Fig. 10(b), Table 1 (AlexNet row) | `scripts/cifar10/alexnet_baseline.sh`, `scripts/cifar10/alexnet_pdt.sh` | lr 0.05, cosine schedule to 1e-3, 60 epochs |
| Fig. 10(a), Table 1 (FCN row) | `scripts/cifar10/fcn_baseline.sh`, `scripts/cifar10/fcn_pdt.sh` | lr 0.01, 100 epochs, `tau = 3` |
| Table 2: base optimizers | `OPTIMIZER={sgd,sgd_m,adam,adamw,rmsprop,shampoo,lamb} scripts/cifar10/optimizers.sh` | learning rates as in the paper; Shampoo and LAMB need `torch-optimizer` |
| Table 4: learning rate x batch size | `LR=<lr> BATCH=<bs> scripts/cifar10/lr_batch_sweep.sh` | no schedule, 60 epochs |
| Table 5, Table 11, Fig. 16, Fig. 17: learning rates with cosine schedule | `LR=<lr> scripts/cifar10/lr_sweep_cosine.sh` | logs gradient norm / direction stability (`--log_gradient_metrics`) |
| Table 6: SGD, momentum, Adam | `OPTIMIZER={sgd,sgd_m,adam} scripts/cifar10/optimizers.sh` | |
| Table 7: profiling | `scripts/cifar10/profiling.sh` | memory / timing / FLOP metrics are printed per epoch (and logged to wandb with `--use_wandb`) |
| Table 10, Fig. 14, Fig. 15: masking ablation | `MASK_MODE={both,accel_only,consistency_only} TAU={3,5} LR={0.01,0.05} scripts/cifar10/masking_ablation.sh` | baseline: `scripts/cifar10/alexnet_baseline.sh` with the same `LR` |
| Table 12, Fig. 18: non-i.i.d. mini-batches | `scripts/cifar10/non_iid.sh` | one class per batch, batch 128 |
| Fig. 5: random acceleration | `scripts/cifar10/random_accelerated.sh` | random subset ratio 0.18, `MASK_SEED` selects the subset |
| Fig. 6: random mask prediction | `scripts/cifar10/random_mask.sh` | random subset ratio 0.097 |
| Fig. 7: switching on validation loss | `scripts/cifar10/switch_by_loss.sh` | |
| Fig. 11: PDT hyper-parameters | `scripts/cifar10/alexnet_pdt.sh --predicted_num <tau> --predict_epoch_interval <T_i> --predict_start_epoch <T_0> --n_past_weights <h>` | lr 0.01 or 0.05 without schedule, 30 epochs, see the figure caption |
| Fig. 12, Fig. 13: mask distribution | `scripts/cifar10/alexnet_pdt.sh --save_masks` | masks are written to `<log_dir>/masks/` |

## ImageNet-1k (multi-GPU)

| Paper | Script | Notes |
|---|---|---|
| Fig. 10(c): ResNet-50 | `scripts/imagenet/resnet50_baseline.sh`, `scripts/imagenet/resnet50_pdt.sh` | SGD with momentum, 600 images per GPU, two-stage cosine schedule, 300 epochs |
| Fig. 10(d): ViT-Base | `scripts/imagenet/vit_base_baseline.sh`, `scripts/imagenet/vit_base_pdt.sh` | AdamW, 600 images per GPU, 200 epochs |
| ViT-Huge | `scripts/imagenet/vit_huge_baseline.sh`, `scripts/imagenet/vit_huge_pdt_layerwise.sh` | AdamW, 128 images per training GPU; GPUs and nodes are set through `NGPUS` and `NNODES`; PDT-Layerwise (one DMD per layer group) |
| Table 8, Table 9: profiling | any ImageNet script with `--count_flops_fast --profile_memory --profile_timing` on a single GPU | metrics are printed per epoch |

The distributed PDT trainers (`pdt_distributed`, `pdt_layerwise`) include the acceleration
scheduler that reduces `tau` and increases `T_i` when the training loss goes up
(`--no_adaptive_schedule` disables it).

## Self-supervised learning and NLP

| Paper | Script | Notes |
|---|---|---|
| Table 3: SimSiam on CIFAR-10 | `scripts/ssl/simsiam_baseline.sh`, `scripts/ssl/simsiam_pdt.sh` | ResNet-18 backbone, 200 epochs, linear-probe accuracy every 10 epochs |
| Table 13, Fig. 19: AG News with a 4-layer LSTM | `scripts/nlp/ag_news_lstm_baseline.sh`, `scripts/nlp/ag_news_lstm_pdt.sh` | dataset downloaded through HuggingFace `datasets` |

## Computing the metrics of the paper from `metrics.csv`

* **Final accuracy**: `accuracy` of the last row.
* **Best training loss**: minimum of `train_loss`.
* **Time to baseline best loss (TTB-Loss)**: let `L*` be the baseline's best (minimum)
  training loss. For each run, TTB-Loss is the `elapsed_s` of the first row whose
  `train_loss` is at or below `L*`; for the baseline this is the epoch in which it reaches
  its own minimum, for PDT it is undefined if PDT never reaches `L*`.
* **Time to baseline best accuracy (TTB-Acc)**: the same rule with the baseline's best
  (maximum) `accuracy`: the `elapsed_s` of the first row whose `accuracy` is at or above
  it.
* **Runtime reduction**: `1 - TTB(PDT) / TTB(baseline)`.
* **Mask ratio**: `mask_ratio` in the rows with `prediction_epoch = 1`.

The aggregation behind the reported results used the baseline run's total runtime as the
baseline TTB-Loss. This file uses the threshold-crossing definition above for both
methods, so values recomputed from new logs can differ slightly from that aggregation.

`elapsed_s` is wall-clock time since the start of training and includes evaluation and
all PDT overheads (snapshot storage, SVD, prediction and masking).
