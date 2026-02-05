# swm-kit

toolkit that automates Stable-WorldModel workflows: record data, train models, and evaluate planners via simple CLI for `World Models`. 

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
