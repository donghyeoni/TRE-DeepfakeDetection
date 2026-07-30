"""Datasets, loaders and transforms for images and precomputed TRE features."""

import glob
import os

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.transforms.functional import to_pil_image, to_tensor

from .. import config


# --------------------------------------------------------------------------- #
# Transforms
# --------------------------------------------------------------------------- #
class CropOrStitch:
    """Crop to ``target_size``; if the image is smaller, tile it first.

    Small images are stitched (tiled) up to ``target_size`` and then cropped
    (random crop for training, center crop otherwise).
    """

    def __init__(self, target_size, train=False):
        self.target_size = target_size
        self.train = train

    def __call__(self, image):
        if isinstance(image, torch.Tensor):
            image = to_pil_image(image)

        width, height = image.size

        if width < self.target_size or height < self.target_size:
            repeat_h = (self.target_size // height) + 1
            repeat_w = (self.target_size // width) + 1

            stitched = torch.cat(
                [torch.cat([to_tensor(image)] * repeat_w, dim=2)] * repeat_h, dim=1
            )
            image = to_pil_image(stitched[:, : self.target_size, : self.target_size])

        if self.train:
            crop_transform = transforms.RandomCrop(self.target_size)
        else:
            crop_transform = transforms.CenterCrop(self.target_size)

        return crop_transform(image)


train_transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
)

test_transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ]
)


# --------------------------------------------------------------------------- #
# Raw image datasets (GenImage / ForenSynths ImageFolder layout)
# --------------------------------------------------------------------------- #
def image_folder(path, train=False):
    """Build a torchvision ``ImageFolder`` with the appropriate transform."""
    return datasets.ImageFolder(str(path), transform=train_transform if train else test_transform)


def build_image_loaders(genimage_root=config.GENIMAGE_ROOT, batch_size=config.BATCH_SIZE):
    """Return ``(train_loader, {generator: val_loader})`` over the raw images.

    Training uses the ``TRAIN_GENERATOR`` train split; every generator's ``val``
    split is used for testing.
    """
    from torch.utils.data import DataLoader

    genimage_root = config.GENIMAGE_ROOT if genimage_root is None else genimage_root
    # GenImage stores sdv4/sdv5 as "sdv1.4"/"sdv1.5" on disk.
    disk_name = {"sdv4": "sdv1.4", "sdv5": "sdv1.5"}

    train_dir = os.path.join(str(genimage_root), disk_name.get(config.TRAIN_GENERATOR, config.TRAIN_GENERATOR), "train")
    train_set = image_folder(train_dir, train=True)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    val_loaders = {}
    for gen in config.GENERATORS:
        val_dir = os.path.join(str(genimage_root), disk_name.get(gen, gen), "val")
        val_loaders[gen] = DataLoader(image_folder(val_dir), batch_size=batch_size, shuffle=False)

    return train_loader, val_loaders


# --------------------------------------------------------------------------- #
# Precomputed TRE feature datasets (.pt tensors)
# --------------------------------------------------------------------------- #
def pt_loader(path):
    """Loader for ``DatasetFolder`` that reads a saved ``.pt`` tensor."""
    return torch.load(path, weights_only=True)


def feature_folder(path):
    """Build a ``DatasetFolder`` over saved ``.pt`` TRE features."""
    return datasets.DatasetFolder(str(path), loader=pt_loader, extensions=(".pt",))


def build_feature_loaders(feature_root=config.TRE_FEATURE_ROOT, batch_size=config.BATCH_SIZE):
    """Return ``(train_loader, {generator: test_loader})`` over the .pt features.

    Expects ``<feature_root>/train`` and ``<feature_root>/test/<generator>``.
    """
    from torch.utils.data import DataLoader

    feature_root = str(feature_root)
    train_set = feature_folder(os.path.join(feature_root, "train"))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    test_loaders = {}
    for gen in config.GENERATORS:
        gen_dir = os.path.join(feature_root, "test", gen)
        if os.path.isdir(gen_dir):
            test_loaders[gen] = DataLoader(feature_folder(gen_dir), batch_size=batch_size, shuffle=False)

    return train_loader, test_loaders


class LatentDiffDataset(Dataset):
    """Dataset over ``.pt`` files, each holding ``{"x": [T, C, H, W], "label": int}``."""

    def __init__(self, root_dir):
        super().__init__()
        self.files = sorted(glob.glob(os.path.join(root_dir, "*.pt")))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx])
        x = data["x"]           # [T, C, H, W] -- step_diffs
        label = data["label"]   # int class
        return x, label
