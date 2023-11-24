import glob, os, shutil, nibabel, csv
import numpy as np
from sklearn.model_selection import KFold

root_dir = '/data/amciilab/jay/datasets/ried/'

oasis_av45s = np.asarray(sorted(glob.glob(os.path.join(root_dir, "OASIS-CAPIIO/*_AV45SUVR_on_CAPIIO.nii"))))
oasis_pibs = np.asarray(sorted(glob.glob(os.path.join(root_dir, "OASIS-CAPIIO/*_PIBSUVR_on_CAPIIO.nii"))))

kf = KFold(n_splits=10)
counter = 0
for train_index, val_index in kf.split(oasis_av45s):
    counter += 1
    
    X_train, X_val = av45s[train_index], av45s[val_index]
    y_train, y_val = pibs[train_index], pibs[val_index]

    if os.path.exists(f"./fold{counter}"):
        shutil.rmtree(f"./fold{counter}")
        print(f"removed an existing ./fold{counter} directory")

    for x in ["train", "val"]:
        if not os.path.exists(f"./fold{counter}/{x}"):
            os.makedirs(f"./fold{counter}/{x}")
        for z in ["images", "targets"]:
            if not os.path.exists(f"./fold{counter}/{x}/{z}"):
                os.makedirs(f"./fold{counter}/{x}/{z}")

    for x in X_train:
        name = x.split("/")[-1]
        img = nibabel.load(x)
        try: 
            data = img.get_fdata()
            os.symlink(x, f"./fold{counter}/train/images/{name}")
        except:
            print("Corrupted file")
    for x in X_val:
        name = x.split("/")[-1]
        img = nibabel.load(x)
        try: 
            data = img.get_fdata()
            os.symlink(x, f"./fold{counter}/val/images/{name}")
        except:
            print("Corrupted file")
    for x in y_train:
        name = x.split("/")[-1]
        img = nibabel.load(x)
        try: 
            data = img.get_fdata()
            os.symlink(x, f"./fold{counter}/train/targets/{name}")
        except:
            print("Corrupted file")
    for x in y_val:
        name = x.split("/")[-1]
        img = nibabel.load(x)
        try: 
            data = img.get_fdata()
            os.symlink(x, f"./fold{counter}/val/targets/{name}")
        except:
            print("Corrupted file")
