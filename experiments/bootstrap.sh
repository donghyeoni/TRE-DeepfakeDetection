#!/bin/bash
# Bring a fresh GPU box to the point where extraction can start.
#
#   bash bootstrap.sh                 # env + repo + test images (25 GB)
#   WITH_TRAIN=1 bash bootstrap.sh    # also the sdv1.4 training archive (96 GB)
#
# Overridable: TRE_HOME (default ~/tre), PYTHON (default python3), WITH_TRAIN.
# Everything downstream reads $TRE_HOME, so nothing else needs editing.
set -e

TRE_HOME="${TRE_HOME:-$HOME/tre}"
PYTHON="${PYTHON:-python3}"
WITH_TRAIN="${WITH_TRAIN:-0}"
REPO_URL="https://github.com/donghyeoni/tre-deepfake-detection"

log(){ echo "[bootstrap $(date +%H:%M:%S)] $1"; }
mkdir -p "$TRE_HOME"
cd "$TRE_HOME"

# --- code -------------------------------------------------------------------
if [ -d repo/.git ]; then
  log "repo present, pulling"
  (cd repo && git pull -q)
else
  log "cloning $REPO_URL"
  git clone -q "$REPO_URL" repo
fi
cp repo/experiments/*.py .

# --- environment ------------------------------------------------------------
# numpy<2 (np.trapz), diffusers 0.31 + transformers 4.44 (older diffusers calls
# huggingface_hub.cached_download; newer transformers needs torch.library.custom_op
# which torch 2.3.1 lacks). See docs/REPRODUCTION.md section 1.
if [ ! -x venv/bin/python ]; then
  log "creating venv"
  "$PYTHON" -m venv venv
  ./venv/bin/pip install -q -U pip
  log "installing torch 2.3.1+cu121 (this is the slow part)"
  ./venv/bin/pip install -q torch==2.3.1 torchvision==0.18.1 \
      --index-url https://download.pytorch.org/whl/cu121
  ./venv/bin/pip install -q "numpy<2" diffusers==0.31.0 transformers==4.44.2 \
      accelerate safetensors huggingface_hub hf_transfer
fi
./venv/bin/python -c "import torch, diffusers, transformers, numpy; \
print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), \
'| diffusers', diffusers.__version__, '| numpy', numpy.__version__)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# --- data -------------------------------------------------------------------
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p data

if [ ! -f data/.test_ready ]; then
  log "downloading the test archive (25 GB)"
  ./venv/bin/python - <<'PY'
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="jzousz/GenImage", repo_type="dataset",
                filename="genimage_test.zip",
                local_dir=os.path.join(os.environ["TRE_HOME"], "data", "hf"))
PY
  log "unzipping"
  ./venv/bin/python -c "import zipfile, os; \
zipfile.ZipFile(os.path.join('$TRE_HOME','data','hf','genimage_test.zip')).extractall(os.path.join('$TRE_HOME','data','test_images'))"
  touch data/.test_ready
fi
log "test images: $(find data/test_images -type f | wc -l)"

if [ "$WITH_TRAIN" = "1" ] && [ ! -f data/.train_ready ]; then
  log "fetching 7-Zip (the training archive is a 30-part zip; unzip cannot open it)"
  mkdir -p bin
  if [ ! -x bin/7zz ]; then
    (cd bin && wget -q https://github.com/ip7z/7zip/releases/download/24.09/7z2409-linux-x64.tar.xz \
      && tar -xf 7z2409-linux-x64.tar.xz 7zz && rm 7z2409-linux-x64.tar.xz)
  fi
  log "downloading the sdv1.4 training archive (96 GB)"
  ./venv/bin/python - <<'PY'
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from huggingface_hub import hf_hub_download
dest = os.path.join(os.environ["TRE_HOME"], "data", "hf")
for i in range(1, 30):
    hf_hub_download(repo_id="jzousz/GenImage", repo_type="dataset",
                    filename=f"stable_diffusion_v_1_4/imagenet_ai_0419_sdv4.z{i:02d}",
                    local_dir=dest)
hf_hub_download(repo_id="jzousz/GenImage", repo_type="dataset",
                filename="stable_diffusion_v_1_4/imagenet_ai_0419_sdv4.zip",
                local_dir=dest)
PY
  log "extracting"
  mkdir -p data/sdv4
  ./bin/7zz x -y -odata/sdv4 data/hf/stable_diffusion_v_1_4/imagenet_ai_0419_sdv4.zip > /dev/null
  touch data/.train_ready
  log "train images: $(find data/sdv4 -type f | wc -l)"
fi

# --- lists ------------------------------------------------------------------
log "building file lists"
TRE_HOME="$TRE_HOME" ./venv/bin/python build_lists.py

cat <<EOF

[bootstrap] ready. TRE_HOME=$TRE_HOME

Next, per GPU k of n:
  export TRE_HOME=$TRE_HOME CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=k
  ./venv/bin/python extract_tre.py --list data/lists/test_sdv4.txt \\
      --out features_eta0/test/sdv4 --batch 48 --eta0 --shard k --nshards n

Roughly 0.78 img/s per L40S; 160k images is ~14 h on four.
Existing outputs are skipped, so runs resume and shards can be re-split.
EOF
