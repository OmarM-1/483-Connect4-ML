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

.PHONY: help venv install setup check-data dataset train clean

help:
	@echo "Available targets:"
	@echo "  make setup       Create virtualenv and install dependencies"
	@echo "  make check-data  Run data inspection script"
	@echo "  make dataset     Run dataset feature build smoke test"
	@echo "  make train       Run baseline training script"
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

clean:
	$(PYTHON) -c "import pathlib, shutil; root=pathlib.Path('.'); shutil.rmtree(root/'.venv', ignore_errors=True); [shutil.rmtree(p, ignore_errors=True) for p in root.rglob('__pycache__')]"
