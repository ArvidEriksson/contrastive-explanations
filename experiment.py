import argparse
parser = argparse.ArgumentParser(description = "Template")

parser.add_argument("-gpu", "--GPU_index", default = 0, type = int, help = "gpu index")
parser.add_argument("-thre", "--threshold", default = 0.4, type = float, help = "the threshold for the second possible class")
parser.add_argument("-model", "--model_name", default = "vgg", type = str, help = "the name of the model")
parser.add_argument("-root", "--data_root", type = str, help = "the root of the dataset")
parser.add_argument("-exp", "--explanation_method", default = "GC", type = str, help = "which explanation method to use out of GC, LA, and XC") 
parser.add_argument("--test_all", default=False, action="store_true", help = "calculate all combinations of negative, ind, and base, overrides their respective arguments")
parser.add_argument("--negative", default=False, action="store_true")
parser.add_argument("-ind", "--index", default = 1, type = int, help = "which index (1 or 2) to calculate the relative score for")
parser.add_argument("-base", "--baseline", default = "blur", type = str, help = "which baseline to use: blur, zero, or mean")

options = parser.parse_args()


import torch
import torchvision
import numpy as np
from torch import nn
from dataset import *
from utils import *

torch.manual_seed(0)
device=torch.device(f'cuda:{options.GPU_index}')


def experiment(samples, testset, model, index=1, mode='positive', baseline='blur', exp_method = 'GC'):
    OriginalProb = []
    ContrastiveProb = []
    Prob = []
    from torchvision import transforms
    GaussianBlur = transforms.GaussianBlur(101, sigma=(10, 20))
    for (n, y1, y2) in tqdm(samples):
        image = testset[n][0].view(1,3,224,224).to(device)

        if index == 1:
            t = y1
        elif index == 2:
            t = y2

        with torch.no_grad():
            pred = model(image)[0]
            prob = torch.softmax(pred, dim = 0)

        saliency_y = get_saliency(model, image, t, mode=exp_method, explanation='original')
        saliency_p = get_saliency(model, image, t, mode=exp_method, explanation='weighted')

        y_images, p_images = equal_blur(image, saliency_y, saliency_p, mode=mode, baseline=baseline)


        with torch.no_grad():
            pred_p = model(p_images)[0]
            prob_p = torch.softmax(pred_p, dim = 0)
            pred_y = model(y_images)[0]
            prob_y = torch.softmax(pred_y, dim = 0)

        OriginalProb.append([torch.exp(pred[y1]).item(), torch.exp(pred[y2]).item()])
        ContrastiveProb.append([torch.exp(pred_p[y1]).item(), torch.exp(pred_p[y2]).item()])
        Prob.append([torch.exp(pred_y[y1]).item(), torch.exp(pred_y[y2]).item()])
    
    OriginalProb = np.array(OriginalProb)
    ContrastiveProb = np.array(ContrastiveProb)
    Prob = np.array(Prob)
    return OriginalProb, ContrastiveProb, Prob

if __name__ == '__main__':
    if options.model_name == 'vgg':
        model = torchvision.models.vgg16_bn(pretrained = False).to(device)
    elif model_name == 'alexnet':
        options.model = torchvision.models.alexnet(pretrained = False).to(device)
    model.classifier[6] = nn.Linear(4096, 200).to(device)
    model.load_state_dict(torch.load(f'model/{options.model_name}_CUB.pth'))

    model.eval()
    testset = CUB(options.data_root, normalization=True, train_test='test')

    samples = get_samples(testset,model,options.threshold)
                    
    mode_map = {"positive" : "higher",
                "negative" : "lower"}
    
    exp_method = options.explanation_method
                    
    if options.test_all:
        for baseline in ["blur", "zero", "mean"]:
            for mode in ["positive", "negative"]:
                    for index in [1, 2]:
                            OriginalProb, ContrastiveProb, Prob = experiment(samples, testset, model, index=index, mode=mode, baseline=baseline, exp_method=exp_method)
                            print("Baseline: %s, Target index: %d, Explanation method: %s"%(baseline, index, exp_method))
                            print("The mode is %s, so the relative score should be the %s the better"%(mode, mode_map[mode]))
                            print("Original\t\t r=%.4f"%((OriginalProb[:,index-1]/OriginalProb.sum(1)).mean()))
                            print("Contrastive\t\t r=%.4f"%((ContrastiveProb[:,index-1]/ContrastiveProb.sum(1)).mean()))
                            print("Blurred\t\t\t r=%.4f"%((Prob[:,index-1]/Prob.sum(1)).mean()))
                        
                    
    else:
        index = options.index
        baseline = options.baseline

        if not options.negative:
            OriginalProb, ContrastiveProb, Prob = experiment(samples, testset, model, index=index, mode='positive', baseline=baseline)
            mode = 'positive'
        else:
            OriginalProb, ContrastiveProb, Prob = experiment(samples, testset, model, index=index, mode='negative', baseline=baseline)
            mode = 'negative'

        print("Baseline: %s, Target index: %d, Explanation method: %s"%(baseline, index, exp_method))
        print("The mode is %s, so the relative score should be the %s the better"%(mode, mode_map[mode]))
        print("Original\t\t r=%.4f"%((OriginalProb[:,index-1]/OriginalProb.sum(1)).mean()))
        print("Contrastive\t r=%.4f"%((ContrastiveProb[:,index-1]/ContrastiveProb.sum(1)).mean()))
        print("Blurred\t\t\t r=%.4f"%((Prob[:,index-1]/Prob.sum(1)).mean()))
