# Reproducibility review of "“Why Not Other Classes?”: Towards Class-Contrastive Back-Propagation Explanations"

This is a repository for the corresponding reproducibility review of the paper "“Why Not Other Classes?”: Towards Class-Contrastive Back-Propagation Explanations".

<!-- Read our report [here](Report.pdf). -->

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

#### Figure 1 and 2
Takes roughly 70 hours to run on a single T4 GPU on Google Cloud VM.
Requires the ImageNet validation dataset to be placed in '../ILSVRC2012'.

```bash
python grad_sign_perturbation.py
```

#### Figure 3
Use the [visualizations.ipynb](visualizations.ipynb) notebook. Requires model fine-tuned on CUB-200.

#### Figure 4
Use the [comparison_max_mean.ipynb](comparison_max_mean.ipynb) notebook. Requires model fine-tuned on CUB-200.

#### Table 1
```
python experiment.py --root <path to unzipped CUB-200> -thre 0.1 -model vgg -exp <GC / LA / XC> --test_all
```
#### Figure 5
The images are generated using the code in [gradcam_vit.ipynb](gradcam_vit.ipynb) (a) and [rollout_vit.ipynb](rollout_vit.ipynb) (b) respectively. 

For both files you need to have downloaded the Mini-Imagenet dataset from Kaggle: https://www.kaggle.com/datasets/deeptrial/miniimagenet/data

To run [gradcam_vit.ipynb](gradcam_vit.ipynb) you need to install `pytorch_grad_cam` using:

```bash
pip install pytorch-gradcam
``` 

To run `rollout_vit.ipynb` you need to clone `vit-explain` to your directory:

```bash
git clone https://github.com/jacobgil/vit-explain.git
```


#### Figure 6
Use the [logit_explanation_relation.ipynb](logit_explanation_relation.ipynb) notebook. Requires the ImageNet validation dataset to be placed in '../ILSVRC2012'.
