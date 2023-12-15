# Initialization

import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt

device = "cuda:0" if torch.cuda.is_available() else "cpu" # If using GPU then use mixed precision training.

# Load ImageNet

data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

normalization = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

data_dir = '../ILSVRC2012'
val_dataset = datasets.ImageNet(root=data_dir, split='val', transform=data_transforms)

BATCH_SIZE = 64

dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
dataset_sizes = len(val_dataset)
class_names = val_dataset.classes

available_models = {
    'resnet18' : torchvision.models.resnet18,
    'alexnet' : torchvision.models.alexnet,
    'googlenet' : torchvision.models.googlenet,
    'mobilenet_v3_small' : torchvision.models.mobilenet_v3_small,
    'mobilenet_v3_large' : torchvision.models.mobilenet_v3_large,
    'mnasnet1_0' : torchvision.models.mnasnet1_0,
    'vgg16' : torchvision.models.vgg16,
    'efficientnet_b1' : torchvision.models.efficientnet_b1,
    'densenet161' : torchvision.models.densenet161
}

def get_model(model_type):

    # Check if the specified model name is valid
    if model_type not in available_models:
        raise ValueError(f"Invalid model name. Available models: {list(available_models.keys())}")
    # Load the model with pre-trained weights
    model = available_models[model_type](weights="DEFAULT").to(device)

    return model

def evaluate_model(model, images_no_norm, labels):
    
    images = normalization(images_no_norm.clone().detach())
    
    was_training = model.training
    model.eval()
    num_images = images.size()[0]

    with torch.no_grad():
        outputs = model(images)
        
        yt_average = outputs[torch.arange(outputs.size(0)), labels].mean().item()
        
        outputs_pt = torch.softmax(outputs, dim=1)
        pt_average = outputs_pt[torch.arange(outputs_pt.size(0)), labels].mean().item()
        
        _, preds = torch.max(outputs, 1)
        correct_classifications = (preds == labels).sum().item()
        accuracy = correct_classifications / num_images

        return accuracy, yt_average, pt_average


def get_explanation(model, inputs_no_norm, targets, explanation_mode):
    model.eval()

    inputs = normalization(inputs_no_norm.clone().detach())

    inputs.requires_grad = True
    model.zero_grad()

    outputs = model(inputs)

    if explanation_mode == 'normal':
        loss = outputs[torch.arange(outputs.size(0)), targets ].sum()
    elif explanation_mode == 'max':
        probs, preds = torch.topk(outputs, 2)
        correct_classifications = preds[:, 0] == targets
        wrong_classifications = preds[:, 0] != targets
        max_index_not_target = correct_classifications * preds[:, 1] + wrong_classifications * preds[:, 0]
        loss = ( outputs[torch.arange(outputs.size(0)), targets ] - outputs[torch.arange(outputs.size(0)), max_index_not_target ] ).sum()
    elif explanation_mode == 'weighted':
        outputs = torch.softmax(outputs, dim=1)
        loss = outputs[torch.arange(outputs.size(0)), targets ].sum()
    elif explanation_mode == 'mean':
        weights = -torch.ones(outputs.shape).to(outputs.device)/999
        weights[range(len(weights)), targets] = 1
        loss = (weights * outputs).sum()

    loss.backward()
    gradients = inputs.grad.clone().detach()
    explanations = gradients
    inputs.requires_grad = False

    return explanations


def grad_sign_perturb(model, inputs_no_norm, targets, no_iterations, eps, explanation_mode):
    x_0 = inputs_no_norm
    x_i = inputs_no_norm
    alpha = eps / no_iterations

    for i in range(no_iterations):
        explanations = get_explanation(model, x_i, targets, explanation_mode)

        # gradients on shape [labels, batch, ...input gradients...]
        
        x_i = x_i + alpha * torch.sign(explanations)
        x_i = torch.clamp(x_i, x_0-eps, x_0+eps)
        x_i = torch.clamp(x_i, 0, 1)

    return x_i


def experiment(dataloader, device, model, no_iterations, no_batches = None, eps=0.001 ):
    
    accuracy_original_total = 0
    accuracy_perturbed_normal_total = 0
    accuracy_perturbed_max_total = 0
    accuracy_perturbed_weighted_total = 0
    accuracy_perturbed_mean_total = 0
    
    yt_original_total = 0
    yt_perturbed_normal_total = 0
    yt_perturbed_max_total = 0
    yt_perturbed_weighted_total = 0
    yt_perturbed_mean_total = 0
    
    pt_original_total = 0
    pt_perturbed_normal_total = 0
    pt_perturbed_max_total = 0
    pt_perturbed_weighted_total = 0
    pt_perturbed_mean_total = 0   

    batch_iter = 0
    for batch in iter(dataloader):
        if no_batches != None and batch_iter >= no_batches:
            break
        batch_iter += 1
        
        inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)

        perturbed_inputs_normal = grad_sign_perturb(model, inputs, targets, no_iterations, eps, 'normal')
        perturbed_inputs_max = grad_sign_perturb(model, inputs, targets, no_iterations, eps, 'max')
        perturbed_inputs_weighted = grad_sign_perturb(model, inputs, targets, no_iterations, eps, 'weighted')
        perturbed_inputs_mean = grad_sign_perturb(model, inputs, targets, no_iterations, eps, 'mean')

        accuracy_original, yt_original, pt_original = evaluate_model(model, inputs, targets)
        accuracy_perturbed_normal, yt_perturbed_normal, pt_perturbed_normal = evaluate_model(model, perturbed_inputs_normal, targets)
        accuracy_perturbed_max, yt_perturbed_max, pt_perturbed_max = evaluate_model(model, perturbed_inputs_max, targets)
        accuracy_perturbed_weighted, yt_perturbed_weighted, pt_perturbed_weighted = evaluate_model(model, perturbed_inputs_weighted, targets)
        accuracy_perturbed_mean, yt_perturbed_mean, pt_perturbed_mean = evaluate_model(model, perturbed_inputs_mean, targets)
        

        accuracy_original_total += accuracy_original
        accuracy_perturbed_normal_total += accuracy_perturbed_normal
        accuracy_perturbed_max_total += accuracy_perturbed_max
        accuracy_perturbed_weighted_total += accuracy_perturbed_weighted
        accuracy_perturbed_mean_total += accuracy_perturbed_mean
        
        yt_original_total += yt_original
        yt_perturbed_normal_total += yt_perturbed_normal
        yt_perturbed_max_total += yt_perturbed_max
        yt_perturbed_weighted_total += yt_perturbed_weighted
        yt_perturbed_mean_total += yt_perturbed_mean
        
        pt_original_total += pt_original
        pt_perturbed_normal_total += pt_perturbed_normal
        pt_perturbed_max_total += pt_perturbed_max
        pt_perturbed_weighted_total += pt_perturbed_weighted
        pt_perturbed_mean_total += pt_perturbed_mean
        
        if (batch_iter % 50 == 0):
            print("batch: ", batch_iter)


    accuracy_original = accuracy_original_total / batch_iter
    accuracy_perturbed_normal = accuracy_perturbed_normal_total / batch_iter
    accuracy_perturbed_max = accuracy_perturbed_max_total / batch_iter
    accuracy_perturbed_weighted = accuracy_perturbed_weighted_total / batch_iter
    accuracy_perturbed_mean = accuracy_perturbed_mean_total / batch_iter
    
    yt_original = yt_original_total / batch_iter
    yt_perturbed_normal = yt_perturbed_normal_total / batch_iter
    yt_perturbed_max = yt_perturbed_max_total / batch_iter
    yt_perturbed_weighted = yt_perturbed_weighted_total / batch_iter
    yt_perturbed_mean = yt_perturbed_mean_total / batch_iter
    
    pt_original = pt_original_total / batch_iter
    pt_perturbed_normal = pt_perturbed_normal_total / batch_iter
    pt_perturbed_max = pt_perturbed_max_total / batch_iter
    pt_perturbed_weighted = pt_perturbed_weighted_total / batch_iter
    pt_perturbed_mean = pt_perturbed_mean_total / batch_iter
    

    return accuracy_original, accuracy_perturbed_normal, accuracy_perturbed_max, accuracy_perturbed_weighted, accuracy_perturbed_mean, \
           yt_original, yt_perturbed_normal, yt_perturbed_max, yt_perturbed_weighted, yt_perturbed_mean, \
           pt_original, pt_perturbed_normal, pt_perturbed_max, pt_perturbed_weighted, pt_perturbed_mean


def plot_figure_3(iter1y_t, iter1p_t, iter1acc_t, iter2y_t, iter2p_t, iter2acc_t, iter10y_t, iter10p_t, iter10acc_t, title, epsilon, with_legend=False):

    x = np.arange(4)
    width = 0.07
    distance = 0.035
    
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    iter1acc_t_bar = ax2.bar(x-4*width-distance, iter1acc_t, width, color='#333333')
    iter1p_t_bar = ax2.bar(x-3*width-distance, iter1p_t, width, color='#878787')
    iter1y_t_bar = ax1.bar(x-2*width-distance, iter1y_t, width, color='#c0c0c0')

    iter2acc_t_bar = ax2.bar(x-width, iter2acc_t, width, color='#ff3333')
    iter2p_t_bar = ax2.bar(x, iter2p_t, width, color='#ffa333')
    iter2y_t_bar = ax1.bar(x+width, iter2y_t, width, color='#ffe4c4')

    iter10acc_t_bar = ax2.bar(x+2*width+distance, iter10acc_t, width, color='#3333ff')
    iter10p_t_bar = ax2.bar(x+3*width+distance, iter10p_t, width, color='#6787e7')
    iter10y_t_bar = ax1.bar(x+4*width+distance, iter10y_t, width, color='#b0c4de')

    #ax1.set_xticks(x, ['Weighted', 'Original', 'Mean', 'Max'])
    ax1.set_xticks(x)
    ax1.set_xticklabels(['Weighted', 'Original', 'Mean', 'Max'])
    
    ax1.set_ylabel('Changes over $y_t$ (one shallow bar)')
    ax2.set_ylabel('Changes over $p_t$ and Accuracy (two dark bars)')
    
    
    if with_legend:
        fig.legend([iter1acc_t_bar, iter1p_t_bar, iter1y_t_bar,
                    iter2acc_t_bar, iter2p_t_bar, iter2y_t_bar,
                    iter10acc_t_bar, iter10p_t_bar, iter10y_t_bar],
                   ["iteration 1, accuracy", "iteration 1, avg. $p_t$", "iteration 1, avg. $y_t$",
                    "iteration 2, accuracy", "iteration 2, avg. $p_t$", "iteration 2, avg. $y_t$",
                    "iteration 10, accuracy", "iteration 10, avg. $p_t$", "iteration 10, avg. $y_t$"],
                   loc='upper center', bbox_to_anchor=(0.5, -0.07), fancybox=True, shadow=True, ncol=3)
    
    plt.title(title)
    
    plt.savefig("Results/results_%s_Epsilon_%s.png"%(title, epsilon))
    
    plt.show()
    
    
if __name__ == '__main__':

    for architecture in ['resnet18', 'alexnet', 'googlenet', 'mobilenet_v3_small', 'mobilenet_v3_large', 'mnasnet1_0', 'vgg16', 'efficientnet_b1', 'densenet161']:
        model = get_model(architecture)
        print("--------------------Network: ", architecture, "--------------------")

        for epsilon in [0.001, 0.003]:
            print("----------Epsilon: ", epsilon, "----------")

            print("no_iterations: 1")
            accuracy_original_1, accuracy_perturbed_normal_1, accuracy_perturbed_max_1, accuracy_perturbed_weighted_1, accuracy_perturbed_mean_1, \
            yt_original_1, yt_perturbed_normal_1, yt_perturbed_max_1, yt_perturbed_weighted_1, yt_perturbed_mean_1, \
            pt_original_1, pt_perturbed_normal_1, pt_perturbed_max_1, pt_perturbed_weighted_1, pt_perturbed_mean_1 = experiment(dataloader, device, model, no_iterations=1, eps=epsilon)

            print("no_iterations: 2")
            accuracy_original_2, accuracy_perturbed_normal_2, accuracy_perturbed_max_2, accuracy_perturbed_weighted_2, accuracy_perturbed_mean_2, \
            yt_original_2, yt_perturbed_normal_2, yt_perturbed_max_2, yt_perturbed_weighted_2, yt_perturbed_mean_2, \
            pt_original_2, pt_perturbed_normal_2, pt_perturbed_max_2, pt_perturbed_weighted_2, pt_perturbed_mean_2 = experiment(dataloader, device, model, no_iterations=2, eps=epsilon)

            print("no_iterations: 10")
            accuracy_original_10, accuracy_perturbed_normal_10, accuracy_perturbed_max_10, accuracy_perturbed_weighted_10, accuracy_perturbed_mean_10, \
            yt_original_10, yt_perturbed_normal_10, yt_perturbed_max_10, yt_perturbed_weighted_10, yt_perturbed_mean_10, \
            pt_original_10, pt_perturbed_normal_10, pt_perturbed_max_10, pt_perturbed_weighted_10, pt_perturbed_mean_10 = experiment(dataloader, device, model, no_iterations=10, eps=epsilon)

            
            result_file = "Results/results_%s_Epsilon_%s.txt"%(architecture, epsilon)
            f = open(result_file, "a")

            f.write("accuracy_original_1 = %s\n" %(accuracy_original_1))
            f.write("accuracy_perturbed_normal_1 = %s\n" %(accuracy_perturbed_normal_1))
            f.write("accuracy_perturbed_max_1 = %s\n" %(accuracy_perturbed_max_1))
            f.write("accuracy_perturbed_weighted_1 = %s\n" %(accuracy_perturbed_weighted_1))
            f.write("accuracy_perturbed_mean_1 = %s\n" %(accuracy_perturbed_mean_1))
            
            f.write("accuracy_original_2 = %s\n" %(accuracy_original_2))
            f.write("accuracy_perturbed_normal_2 = %s\n" %(accuracy_perturbed_normal_2))
            f.write("accuracy_perturbed_max_2 = %s\n" %(accuracy_perturbed_max_2))
            f.write("accuracy_perturbed_weighted_2 = %s\n" %(accuracy_perturbed_weighted_2))
            f.write("accuracy_perturbed_mean_2 = %s\n" %(accuracy_perturbed_mean_2))

            f.write("accuracy_original_10 = %s\n" %(accuracy_original_10))
            f.write("accuracy_perturbed_normal_10 = %s\n" %(accuracy_perturbed_normal_10))
            f.write("accuracy_perturbed_max_10 = %s\n" %(accuracy_perturbed_max_10))
            f.write("accuracy_perturbed_weighted_10 = %s\n" %(accuracy_perturbed_weighted_10))
            f.write("accuracy_perturbed_mean_10 = %s\n" %(accuracy_perturbed_mean_10))
            
            
            f.write("yt_original_1 = %s\n" %(yt_original_1))
            f.write("yt_perturbed_normal_1 = %s\n" %(yt_perturbed_normal_1))
            f.write("yt_perturbed_max_1 = %s\n" %(yt_perturbed_max_1))
            f.write("yt_perturbed_weighted_1 = %s\n" %(yt_perturbed_weighted_1))
            f.write("yt_perturbed_mean_1 = %s\n" %(yt_perturbed_mean_1))
            
            f.write("yt_original_2 = %s\n" %(yt_original_2))
            f.write("yt_perturbed_normal_2 = %s\n" %(yt_perturbed_normal_2))
            f.write("yt_perturbed_max_2 = %s\n" %(yt_perturbed_max_2))
            f.write("yt_perturbed_weighted_2 = %s\n" %(yt_perturbed_weighted_2))
            f.write("yt_perturbed_mean_2 = %s\n" %(yt_perturbed_mean_2))

            f.write("yt_original_10 = %s\n" %(yt_original_10))
            f.write("yt_perturbed_normal_10 = %s\n" %(yt_perturbed_normal_10))
            f.write("yt_perturbed_max_10 = %s\n" %(yt_perturbed_max_10))
            f.write("yt_perturbed_weighted_10 = %s\n" %(yt_perturbed_weighted_10))
            f.write("yt_perturbed_mean_10 = %s\n" %(yt_perturbed_mean_10))


            f.write("pt_original_1 = %s\n" %(pt_original_1))
            f.write("pt_perturbed_normal_1 = %s\n" %(pt_perturbed_normal_1))
            f.write("pt_perturbed_max_1 = %s\n" %(pt_perturbed_max_1))
            f.write("pt_perturbed_weighted_1 = %s\n" %(pt_perturbed_weighted_1))
            f.write("pt_perturbed_mean_1 = %s\n" %(pt_perturbed_mean_1))
            
            f.write("pt_original_2 = %s\n" %(pt_original_2))
            f.write("pt_perturbed_normal_2 = %s\n" %(pt_perturbed_normal_2))
            f.write("pt_perturbed_max_2 = %s\n" %(pt_perturbed_max_2))
            f.write("pt_perturbed_weighted_2 = %s\n" %(pt_perturbed_weighted_2))
            f.write("pt_perturbed_mean_2 = %s\n" %(pt_perturbed_mean_2))

            f.write("pt_original_10 = %s\n" %(pt_original_10))
            f.write("pt_perturbed_normal_10 = %s\n" %(pt_perturbed_normal_10))
            f.write("pt_perturbed_max_10 = %s\n" %(pt_perturbed_max_10))
            f.write("pt_perturbed_weighted_10 = %s\n" %(pt_perturbed_weighted_10))
            f.write("pt_perturbed_mean_10 = %s\n" %(pt_perturbed_mean_10))

            f.close()


            # Plot results

            iter1y_t = [yt_perturbed_weighted_1-yt_original_1, yt_perturbed_normal_1-yt_original_1, yt_perturbed_mean_1-yt_original_1, yt_perturbed_max_1-yt_original_1]
            iter1p_t = [pt_perturbed_weighted_1-pt_original_1, pt_perturbed_normal_1-pt_original_1, pt_perturbed_mean_1-pt_original_1, pt_perturbed_max_1-pt_original_1]
            iter1acc_t = [accuracy_perturbed_weighted_1-accuracy_original_1, accuracy_perturbed_normal_1-accuracy_original_1, accuracy_perturbed_mean_1-accuracy_original_1, accuracy_perturbed_max_1-accuracy_original_1]
            iter2y_t = [yt_perturbed_weighted_2-yt_original_2, yt_perturbed_normal_2-yt_original_2, yt_perturbed_mean_2-yt_original_2, yt_perturbed_max_2-yt_original_2]
            iter2p_t = [pt_perturbed_weighted_2-pt_original_2, pt_perturbed_normal_2-pt_original_2, pt_perturbed_mean_2-pt_original_2, pt_perturbed_max_2-pt_original_2]
            iter2acc_t = [accuracy_perturbed_weighted_2-accuracy_original_2, accuracy_perturbed_normal_2-accuracy_original_2, accuracy_perturbed_mean_2-accuracy_original_2, accuracy_perturbed_max_2-accuracy_original_2]
            iter10y_t = [yt_perturbed_weighted_10-yt_original_10, yt_perturbed_normal_10-yt_original_10, yt_perturbed_mean_10-yt_original_10, yt_perturbed_max_10-yt_original_10]
            iter10p_t = [pt_perturbed_weighted_10-pt_original_10, pt_perturbed_normal_10-pt_original_10, pt_perturbed_mean_10-pt_original_10, pt_perturbed_max_10-pt_original_10]
            iter10acc_t = [accuracy_perturbed_weighted_10-accuracy_original_10, accuracy_perturbed_normal_10-accuracy_original_10, accuracy_perturbed_mean_10-accuracy_original_10, accuracy_perturbed_max_10-accuracy_original_10]

            plot_figure_3(iter1y_t, iter1p_t, iter1acc_t, iter2y_t, iter2p_t, iter2acc_t, iter10y_t, iter10p_t, iter10acc_t, architecture, epsilon)
