# swm-kit

Ultraminimal Unix-style helpers for `stable-worldmodel`.

## Install

```bash
pip install -e .
```

## Commands

```bash
swm record --env swm/PushT-v1 --out data.h5 --episodes 100
swm train --data data.h5 --out model.pt --seed 0
swm eval --env swm/PushT-v1 --model model.pt --episodes 10 --seed 0
```

## Full example

```bash
swm record --env swm/PushT-v1 --out data.h5 --episodes 100
swm train --data data.h5 --out model.pt --seed 0
swm eval --env swm/PushT-v1 --model model.pt --episodes 10 --seed 0
```

## Philosophy

Small, simple, composable CLIs: stdin → processing → stdout.
