VENV_DIR := .venv
VENV_CFG := $(VENV_DIR)/pyvenv.cfg
DEPS_STAMP := $(VENV_DIR)/.deps_installed

ifeq ($(OS),Windows_NT)
PYTHON ?= py -3
VENV_PY := $(VENV_DIR)/Scripts/python.exe
VENV_PIP := $(VENV_DIR)/Scripts/pip.exe
else
PYTHON ?= python3
VENV_PY := $(VENV_DIR)/bin/python
VENV_PIP := $(VENV_DIR)/bin/pip
endif

.PHONY: help venv install setup check-data dataset train train-phase train-rl-4-12 train-rf-filtered clean

PHASE ?= mid
PHASE_TRAIN_CSV ?= data/UCI-Midgame-d30.train.csv
PHASE_VAL_CSV ?= data/UCI-Midgame-d30.val.csv
RL412_TRAIN_CSV ?= data/UCI-Midgame-d30.train.csv
RL412_VAL_CSV ?= data/UCI-Midgame-d30.val.csv
RL412_EPISODES ?= 3000
RL412_BATCH ?= 64
RL412_MAX_ROWS_TRAIN ?= 5000
RL412_MAX_ROWS_VAL ?= 1000

help:
	@echo "Available targets:"
	@echo "  make setup       Create virtualenv and install dependencies"
	@echo "  make check-data  Run data inspection script"
	@echo "  make dataset     Run dataset feature build smoke test"
	@echo "  make train       Run baseline training script"
	@echo "  make train-phase Run phase-aware policy training script"
	@echo "  make train-rl-4-12 Run RL training locked to 4-12 ply window"
	@echo "  make clean       Remove virtualenv and Python cache files"
	@echo ""
	@echo "Notes:"
	@echo "  - Works on Linux/macOS and Windows 11 (with GNU Make installed)."
	@echo "  - Override interpreter with: make PYTHON=python3 setup"

venv: $(VENV_CFG)

$(VENV_CFG):
	$(PYTHON) -m venv $(VENV_DIR)

install: $(DEPS_STAMP)

$(DEPS_STAMP): requirements.txt | $(VENV_CFG)
	$(VENV_PIP) install --upgrade pip
	$(VENV_PIP) install -r requirements.txt
	$(VENV_PY) -c "import pathlib; pathlib.Path('$(DEPS_STAMP)').touch()"

setup: install

check-data: install
	$(VENV_PY) src/check_data.py

dataset: install
	$(VENV_PY) src/dataset.py

train: install
	$(VENV_PY) src/train_baseline.py

train-phase: install
	$(VENV_PY) src/train_phase_policy.py \
		--phase $(PHASE) \
		--train-csv $(PHASE_TRAIN_CSV) \
		--val-csv $(PHASE_VAL_CSV)

train-rl-4-12: install
	$(VENV_PY) src/train_phase_policy.py \
		--training-mode rl \
		--rl-backend auto \
		--rl-device auto \
		--rl-policy-arch linear \
		--rl-algo reinforce \
		--rl-seed-weights \
		--phase custom \
		--move-min 4 \
		--move-max 12 \
		--train-csv $(RL412_TRAIN_CSV) \
		--val-csv $(RL412_VAL_CSV) \
		--max-rows-train $(RL412_MAX_ROWS_TRAIN) \
		--max-rows-val $(RL412_MAX_ROWS_VAL) \
		--rl-episodes $(RL412_EPISODES) \
		--rl-batch-episodes $(RL412_BATCH) \
		--rl-lr 0.01 \
		--rl-seed-steps 300 \
		--rl-seed-batch-size 512 \
		--rl-seed-lr 0.01 \
		--rl-eval-interval-episodes 100 \
		--rl-entropy-coef 0.005 \
		--rl-entropy-coef-final 0.0005 \
		--rl-epsilon 0.03 \
		--rl-epsilon-final 0.005 \
		--rl-hard-replay-fraction 0.25 \
		--rl-hard-replay-steps 2 \
		--random-state 42 \
		--model-out artifacts/models/connect4_phase_policy_rl_reinforce_linear_seeded_ply4_12.pkl \
		--metrics-out artifacts/metrics/connect4_phase_policy_rl_reinforce_linear_seeded_ply4_12_metrics.json \
		--report-out artifacts/reports/connect4_phase_policy_rl_reinforce_linear_seeded_ply4_12_report.txt

train-rf-filtered: install
	$(VENV_PY) src/train/train_rf_filtered.py

clean:
	$(PYTHON) -c "import pathlib, shutil; root=pathlib.Path('.'); shutil.rmtree(root/'.venv', ignore_errors=True); [shutil.rmtree(p, ignore_errors=True) for p in root.rglob('__pycache__')]"
