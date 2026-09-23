# PDT: Predictive Differential Training Guided by Training Dynamics

Official code for the ICLR 2026 paper
[*Predictive Differential Training Guided by Training Dynamics*](https://openreview.net/forum?id=zSTgrLkpRi)
by Fanqi Wang\*, Weisheng Tang\*, Landon Harris, Hairong Qi, Dan Wilson and Igor Mezić.

## About this commit

This commit (tag `iclr2026-experiment-code`) is a **sanitized archival collection of the
implementations that were used to run the experiments in the paper**, kept for reference.
It combines the single-GPU research tree and the multi-GPU cluster tree that were in use.
The paper experiments were not produced from a single repository state;
[`HISTORY.md`](HISTORY.md) explains which file produced which experiment and lists every
edit made for publication (private paths and account names replaced, unused code and
internal notes removed; no training or masking logic changed). Third-party attributions
are in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

The cleaned, documented and recommended implementation is on the `main` branch
(tag `v1.0` and later). Please start there.

## License

Apache License 2.0, see [`LICENSE`](LICENSE).
