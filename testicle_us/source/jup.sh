#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --partition=all_serial
#SBATCH --mem=20G
#SBATCH --nodes=1
#SBATCH --account=ai4bio2024
#SBATCH --cpus-per-task=6
#SBATCH -o /work/tesi_nmorelli/testicol_us/.logs/ourjupyter%j.log
#SBATCH -e /work/tesi_nmorelli/testicol_us/.logs/errjupyter%j.log
#SBATCH --nodelist=ailb-login-03

WEBHOOK_URL="https://discord.com/api/webhooks/1349866019438723224/AcEubkRsN0qDEa4MKa626-u5nGhWtL7PMnXYa6aRMV2BzsPsQDEsy4FQzBo290V2ds6h"
LOGFILE="/work/tesi_nmorelli/testicol_us/.logs/ourjupyter${SLURM_JOB_ID}.log"
source ~/.venv/bin/activate


# Start Jupyter notebook in the background
jupyter notebook --no-browser --ip=0.0.0.0 --port=8889 > "$LOGFILE" 2>&1 &

# Wait for the notebook to start and extract the URL
while ! grep -q "http://.*:8889/" "$LOGFILE"; do
    sleep 2
done

URL=$(grep -o "http://.*:8889/[^ ]*" "$LOGFILE" | head -n 1)

# Send to Discord
curl -H "Content-Type: application/json" \
     -X POST \
     -d "{\"content\": \"🔗 Jupyter Notebook is live: $URL\"}" \
     "$WEBHOOK_URL"

# Wait for Jupyter to exit to keep the job running
wait