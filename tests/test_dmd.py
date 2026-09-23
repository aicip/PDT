"""DMD sanity check on a synthetic linear system (runs on CPU in a few seconds).

    python tests/test_dmd.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from kcl.lib.dmd.torchDMD import TorchDMD  # noqa: E402


def main():
    torch.manual_seed(0)
    n, r, steps = 2000, 3, 8
    # low-rank linear dynamics x_{k+1} = A x_k with A = Q diag(lambda) Q^+
    Q = torch.linalg.qr(torch.randn(n, r))[0]
    lam = torch.tensor([0.97, 0.9, 0.8])
    x0 = Q @ torch.randn(r)
    snapshots = [x0]
    for _ in range(steps):
        snapshots.append(Q @ (lam * (Q.T @ snapshots[-1])))
    W = torch.stack(snapshots[:6], dim=1)  # first 6 states as the history

    dmd = TorchDMD(rank=r)
    dmd.fit(W)
    tau = 3
    pred = dmd.predict_multistep(W[:, -1].reshape(-1, 1), tau).view(-1)
    truth = snapshots[5 + tau]
    rel_err = (pred - truth).norm() / truth.norm()
    print(f"eigenvalues (moduli): {sorted(dmd.eigs.abs().tolist(), reverse=True)}")
    print(f"relative error of the {tau}-step prediction: {rel_err:.2e}")
    assert rel_err < 1e-3, "DMD prediction error too large"
    print("test_dmd: OK")


if __name__ == "__main__":
    main()
