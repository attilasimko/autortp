#!/usr/bin/env bash
#SBATCH -A NAISS2024-5-188 -p alvis
#SBATCH -N 1 --gpus-per-node=A100:1
#SBATCH --time=01-00:00:00
#SBATCH --mail-user=attila.simko@umu.se --mail-type=end
#SBATCH --error=/cephyr/users/attilas/Alvis/out/%J_error.out
#SBATCH --output=/cephyr/users/attilas/Alvis/out/%J_output.out

module load TensorFlow/2.11.0-foss-2022a-CUDA-11.7.0
source /cephyr/users/attilas/Alvis/venv/bin/activate

python3 train.py --base alvis
wait