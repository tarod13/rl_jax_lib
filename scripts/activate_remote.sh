#!/bin/bash
module load python/3.11 cuda/12.9
source venv/bin/activate
export XLA_PYTHON_CLIENT_PREALLOCATE=false
echo "Environment activated! Ready to code."