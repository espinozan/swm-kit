# swm-kit

Ultraminimal Unix-style helpers for `stable-worldmodel`: record datasets, train a tiny baseline model, evaluate planners.

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

## Notes

- `record` uses `stable_worldmodel.World.record_dataset`.
- `train` expects `obs`/`observations` and `actions`/`action` datasets.
- `record` and `train` echo the output path; `eval` prints mean reward.

## Philosophy

Small, simple, composable CLIs: stdin → processing → stdout.
