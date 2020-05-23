#!/usr/bin/env bash

conda activate rl

black -l 160 rl-6-nimmt/*.py rl-6-nimmt/agents/*.py rl-6-nimmt/utils/*.py rl-6-nimmt/game/*.py
