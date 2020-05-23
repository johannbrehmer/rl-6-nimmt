#!/usr/bin/env bash

conda activate rl

black -l 160 rl_6_nimmt/*.py rl_6_nimmt/agents/*.py rl_6_nimmt/utils/*.py
