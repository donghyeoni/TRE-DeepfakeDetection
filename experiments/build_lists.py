"""Build extraction list files from the unpacked GenImage data."""
import glob, os, random

BASE = os.environ.get("TRE_DATA",
                      os.path.join(os.environ.get("TRE_HOME", os.path.expanduser("~/tre")), "data"))
GEN_DIRS = {"adm": "adm_imagenet", "biggan": "biggan_imagenet", "glide": "glide_imagenet",
            "midjourney": "midjourney_imagenet", "sdv4": "sdv4_imagenet", "sdv5": "sdv5_imagenet",
            "vqdm": "vqdm_imagenet", "wukong": "wukong_imagenet"}

os.makedirs(f"{BASE}/lists", exist_ok=True)

# ---- train: 30k fake (ai) + 30k real (nature) from sdv4 train split ----
random.seed(42)
train_root = glob.glob(f"{BASE}/sdv4/imagenet_ai_0419_sdv4/train")[0]
ai = sorted(glob.glob(f"{train_root}/ai/*"))
nat = sorted(glob.glob(f"{train_root}/nature/*"))
print("sdv4 train pool:", len(ai), "ai,", len(nat), "nature")
random.shuffle(ai); random.shuffle(nat)
with open(f"{BASE}/lists/train.txt", "w") as f:
    for p in ai[:30000]: f.write(f"{p}\tfake\n")
    for p in nat[:30000]: f.write(f"{p}\treal\n")

# ---- test: every generator's full test split ----
for gen, d in GEN_DIRS.items():
    ai = sorted(glob.glob(f"{BASE}/test_images/test/{d}/ai/*"))
    nat = sorted(glob.glob(f"{BASE}/test_images/test/{d}/nature/*"))
    print(gen, len(ai), "ai,", len(nat), "nature")
    with open(f"{BASE}/lists/test_{gen}.txt", "w") as f:
        for p in ai: f.write(f"{p}\tfake\n")
        for p in nat: f.write(f"{p}\treal\n")
print("LISTS_DONE")
