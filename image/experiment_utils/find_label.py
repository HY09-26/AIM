import numpy as np

idx500 = np.load("data/ImageNet/imagenet_val_500.npy")
orig_idx = int(idx500[475])

print("Original ImageNet val index =", orig_idx)


from torchvision import datasets, transforms

val_dir = "data/ImageNet/val"

tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

dataset = datasets.ImageFolder(val_dir, transform=tf)

img_path, label = dataset.samples[orig_idx]

print("Image path:", img_path)
print("Class index:", label)
print("Synset:", dataset.classes[label])


import scipy.io as sio

meta = sio.loadmat("data/ImageNet/ILSVRC2012_devkit_t12/data/meta.mat")
synsets = meta["synsets"]

synset_to_name = {}
for s in synsets:
    wnid = s[0][1][0]           # e.g. 'n02124075'
    words = s[0][2][0]          # e.g. 'Egyptian cat'
    synset_to_name[wnid] = words

print("Class name:", synset_to_name[dataset.classes[label]])
