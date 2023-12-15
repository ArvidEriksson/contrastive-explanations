# Reproducibility review of "“Why Not Other Classes?”: Towards Class-Contrastive Back-Propagation Explanations"

This is a repository for the corresponding reproducibility review of the paper "“Why Not Other Classes?”: Towards Class-Contrastive Back-Propagation Explanations".

## Content

### Libraries
```
numpy==1.23.5
torch==1.13.1
torchvision=0.14.1
```

### To reproduce the results


#### Training fine-tuned VGG-16 model on CUB-200
```
python train.py --root <path to unzipped CUB-200>
```

#### Figure 3
Use the [visualizations.ipynb](visualizations.ipynb) notebook. Requires model fine-tuned on CUB-200.

#### Figure 4
Use the [comparison_max_mean.ipynb](comparison_max_mean.ipynb) notebook. Requires model fine-tuned on CUB-200.

#### Table 1
```
python experiment.py --root <path to unzipped CUB-200> -thre 0.1 -model vgg -exp <GC / LA / XC> --test_all
```