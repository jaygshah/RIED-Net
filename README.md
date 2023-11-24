# RIED-Net
Official PyTorch implementation of [**Deep Residual Inception Encoder-Decoder Network for Amyloid PET Harmonization**](https://alz-journals.onlinelibrary.wiley.com/doi/10.1002/alz.12564) [Alzheimer's &amp; Dementia]

[Jay Shah](https://www.public.asu.edu/~jgshah1/)<sup>1,2</sup>,
Fei Gao<sup>1,2</sup>, 
[Baoxin Li](https://www.public.asu.edu/~bli24/)<sup>1,2</sup>,
Valentina Ghisays<sup>3</sup>, 
Ji Luo<sup>3</sup>, 
Yinghua Chen<sup>3</sup>, 
Wendy Lee<sup>3</sup>, 
Yuxiang Zhou<sup>4</sup>, 
[Tammie L.S. Benzinger](https://scholar.google.com/citations?user=fr-fkIwAAAAJ&hl=en)<sup>5</sup>, 
[Eric M. Reiman](https://scholar.google.com/citations?user=I-Khl7AAAAAJ&hl=en)<sup>3</sup>,
[Kewei Chen](https://scholar.google.com/citations?user=d83ZIzEAAAAJ&hl=en)<sup>3</sup>,
[Yi Su](https://scholar.google.com/citations?user=vdZKSEIAAAAJ&hl=en)<sup>1,2,3</sup>,
[Teresa Wu](https://labs.engineering.asu.edu/wulab/person/teresa-wu-2/)<sup>1,2</sup>

<sup>1</sup>ASU-Mayo Center for Innovative Imaging,
<sup>2</sup>Arizona State University,
<sup>3</sup>Banner Alzheimer’s Institute,
<sup>4</sup>Dept of Radiology, Mayo Clinic, Arizona,
<sup>5</sup>Mallinckrodt Inst. of Radiology, Washington University

---
Multiple positron emission tomography (PET) tracers are available for amyloid imaging, posing a significant challenge to consensus interpretation and quantitative analysis in Alzheimer's disease research. We accordingly developed and validated a deep learning model as a harmonization strategy. Learn more about its practical applications in ASU's blog post: [Using AI to battle Alzheimer's.](https://news.asu.edu/20220315-solutions-using-ai-battle-alzheimers-asu-researchers-banner-health-team-up)

<p align="center">
<img src="imgs/ried_results.jpg" width=62% height=62% 
class="center">
</p>
Visual comparison of synthetic images generated using RIED-Net to real Pittsburgh Compound-B (PIB) data for the OASIS (A &amp; B) and GAAIN (C &amp; D) datasets

## Installation
Instructions to install MONAI can be found [here](https://docs.monai.io/en/stable/installation.html) and the appropriate version of Pytorch using [locally](https://docs.monai.io/en/stable/installation.html).
Packages used in the current version of this code.
```
monai==1.3.0
torch==2.1.0+cu118
torchaudio==2.1.0+cu118
torchmetrics==1.2.0
torchvision==0.16.0+cu118
```
## Dataset
You should structure your aligned dataset in the following way:
```
trainfold/
  ├── train
    ├──images
      ├──xxx.nii
      ├──...
    ├──targets
      ├──yyy.nii
      ├──...
  ├── val
    ├──images
      ├──xxx.nii
      ├──...
    ├──targets
      ├──yyy.nii
      ├──...
```

## Training 
```
python train.py --dataset trainfold1 --batch_size 3 --model_name resunet
```
## Evaluation
`testfold` should have same structure as `trainfold`. Checkout `prepare_data.py` to prepare the data folds.
```
python generate.py --dataset testfold1 --model_name resunet
```
## Citation
Please consider citing RIED-Net if this repository is useful for your work. 
```
@article{shah2022deep,
  title={Deep residual inception encoder-decoder network for amyloid PET harmonization},
  author={Shah, Jay and Gao, Fei and Li, Baoxin and Ghisays, Valentina and Luo, Ji and Chen, Yinghua and Lee, Wendy and Zhou, Yuxiang and Benzinger, Tammie LS and Reiman, Eric M and others},
  journal={Alzheimer's \& Dementia},
  volume={18},
  number={12},
  pages={2448--2457},
  year={2022},
  publisher={Wiley Online Library}
}
```
## Acknowledgments
This research has been supported partially by NIH grants R01AG031581, R01AG069453, P30AG019610, and Arizona Department of Health Services (ADHS) and the State of Arizona, ADHS Grant No. CTR040636. This is a [patent-pending](https://patentscope.wipo.int/search/en/detail.jsf?docId=WO2023101959&_cid=P21-LPD5OU-06483-1) technology.
