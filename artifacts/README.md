# Model Checkpoints & Cached Tensor Artifacts

Heavy binary checkpoints (`.ckpt`, `.pt`) and per-image prediction arrays (`.npz`) are kept locally under `checkpoints/` / `research/` and excluded from public Git distribution due to file size (> 3.2 GiB total) and derivative `CC-BY-NC 4.0` governance (see `../LICENSE_STATUS.md`).

Expected SHA-256 verification digests (defined in `research/derma_act/config.py`):

| Artifact | Role | Expected SHA-256 |
| :--- | :--- | :--- |
| `epoch=09-val_auroc=0.0000.ckpt` | Frozen EfficientNet-B4 backbone weights | `0d14579d668d02781874cf91ad493fac4bf33ed53e3be928e3a7f9bb856ebbbb` |
| `policy.pt` | Frozen `MultiHeadActionPolicy` (`6,787` params) | `32a098c8e2aa958959122cb0292eff0cc8932ea4dac4c2599f3dd37ed2587c95` |
| `lambda_selection.json` | Utility coefficients `(5.0, 1.0, 0.5, 0.25)` | `3662d80eacb9ca39276697ff5f9d043d3774794a9589f316b81b2c0fdf0dea05` |
| `stopping_threshold.json` | Threshold `tau = -3/7` (`-0.4285714...`) | `dc13a238e16a678bf153beb8b2cbbb7341f8360f8e8d0f8172ee68bf67eaf8ef` |
