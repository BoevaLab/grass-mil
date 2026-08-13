#!/bin/bash
# Schedule execution of many runs
# Run from the repository root with: bash scripts/schedule.sh

grass-mil-train trainer.max_epochs=5 logger=csv

grass-mil-train trainer.max_epochs=10 logger=csv
