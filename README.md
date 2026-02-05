# swm-kit

toolkit that automates workflows: record data, train models, and evaluate planners via simple CLI for Stable-WorldModel an eficient World model research made simple. From data collection to training and evaluation.

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

## Philosophy

Small, simple, composable CLIs: stdin → processing → stdout.
