# swm-kit

Ultraminimal Unix-style helpers for `stable-worldmodel`. No reimplementation, only light wrappers.

## Install

```bash
pip install -e .
```

## Commands

```bash
swm record --env swm/PushT-v1 --out data.h5 --episodes 100
swm train --data data.h5 --out model.pt
swm eval --env swm/PushT-v1 --model model.pt
```

## Full example

```bash
swm record --env swm/PushT-v1 --out data.h5 --episodes 100
swm train --data data.h5 --out model.pt
swm eval --env swm/PushT-v1 --model model.pt
```

## Data contract

- `record` writes an HDF5 file using `stable_worldmodel.World.record_dataset`.
- `train` expects datasets keyed as `obs`/`observations` and `actions`/`action`.
- `eval` loads the saved `.pt` and prints mean reward to stdout.

## Philosophy

Small, simple, composable CLIs: stdin → processing → stdout.
