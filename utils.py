import torch
import torchvision
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms
from tqdm import tqdm

def get_n_params(model):
    n_param = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            n_param += torch.tensor(param.shape).prod()
    print(f'Then number of parameters is {n_param}.')


def normalize(x):
    device = x.device
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    xx = x.clone().detach().to(device)
    xx[:, 0, :, :] = (x[:, 0, :, :] - mean[0]) / std[0] 
    xx[:, 1, :, :] = (x[:, 1, :, :] - mean[1]) / std[1] 
    xx[:, 2, :, :] = (x[:, 2, :, :] - mean[2]) / std[2] 
    return xx

def denorm(x):
    device = x.device
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    xx = x.clone().detach().to(device)
    
    if len(xx.shape) == 4:
        xx[:, 0, :, :] = x[:, 0, :, :] * std[0] + mean[0]
        xx[:, 1, :, :] = x[:, 1, :, :] * std[1] + mean[1]
        xx[:, 2, :, :] = x[:, 2, :, :] * std[2] + mean[2]
    elif len(xx.shape) == 3:
        xx[0, :, :] = x[0, :, :] * std[0] + mean[0]
        xx[1, :, :] = x[1, :, :] * std[1] + mean[1]
        xx[2, :, :] = x[2, :, :] * std[2] + mean[2]
    return xx

def plot_tensor_image(image):
    toshow = image.detach().clone().cpu()
    toshow = toshow.squeeze()
    if toshow.max() > 1:
        toshow = denorm(toshow)
    
    if len(toshow.shape) == 4:
        print("Plotting the entire batch:")
        toshow = torchvision.utils.make_grid(toshow)
        figsize = (20,20)
    else:
        print("Plotting the single image")
        figsize = (5,5)
    
    plt.figure(figsize=figsize)
    plt.imshow(toshow.numpy().transpose((1,2,0)))
    plt.axis('off')
    plt.show()

def get_saliency(
    model, 
    image, 
    label, 
    mode='GC', 
    explanation='contrastive',
    use_relu=False):

    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output#.detach()
        return hook
    
    if 'vgg' in str(type(model)):
        handle = model.features[40].register_forward_hook(get_activation('feature'))
        size = 14
    elif 'alexnet' in str(type(model)):
        handle = model.features[10].register_forward_hook(get_activation('feature'))
        size = 13
    
    if explanation == 'original':
        # loss = outputs[torch.arange(outputs.size(0)), targets ].sum()
        output = model(image)[0, label]
    elif explanation == 'weighted':
        output = torch.softmax(model(image),dim=1)[0, label]
    elif explanation == 'max':
        output = model(image)
        probs, preds = torch.topk(output, 2)
        correct_classifications = preds[0, 0] == label
        wrong_classifications = preds[0, 0] != label
        max_index_not_target = correct_classifications * preds[0, 1] + wrong_classifications * preds[0, 0]
        #loss = ( outputs[torch.arange(outputs.size(0)), targets ] - outputs[torch.arange(outputs.size(0)), preds[:,1] ] ).sum()
        output = (output[0, label] - output[0, max_index_not_target])
    elif explanation == 'mean':
        #loss = ( outputs[torch.arange(outputs.size(0)), targets ] * (1 + 1/( 1 - 1000 )) - torch.sum(outputs[torch.arange(outputs.size(0))]) * 1/(( 1 - 1000 )) ).sum()
        output = model(image)
        weights = -torch.ones(output.shape).to(output.device)/(output.size(1))
        weights[0, label] = 1
        output = (weights * output).sum()
        #output = output[0, label].sum()
        
        #weights = -torch.ones(outputs.shape).to(outputs.device)/999
        #weights[range(len(weights)), targets] = 1
        #loss = (weights * outputs).sum()
        #loss = pred[range(len(pred)), label].sum()
    
        
    handle.remove()
    grad = torch.autograd.grad(
        inputs = activation['feature'],
        outputs = output,
        create_graph = True
    )[0][0]
    activations = activation['feature'].squeeze()
    if mode == 'GC':
        weights = grad.mean(dim=[1,2], keepdim=True)
        saliency = (weights*activations).sum(dim=0)
    elif mode == 'LA':
        # This is the same as HiResCAM
        #saliency = (activation['feature'].squeeze()*grad).mean(0)
        elementwise_activations = grad * activations
        saliency = elementwise_activations.sum(dim=0)
    elif mode == 'XC':
        sum_activations = (activations.abs().sum(dim=[1,2], keepdim=True))
        eps = 1e-7
        weights = grad * activations / (sum_activations + eps)
        weights = weights.sum(dim=[1,2], keepdim=True)
        saliency = (weights*activations).sum(dim=0)
    else:
        raise ValueError('Incorrect mode given to get_saliency.')
    return F.interpolate(saliency.view(1,1,size,size), (224, 224), mode = 'bilinear', align_corners = True).squeeze()

def blur(image, saliency, mode='positive'):
    
    GaussianBlur = transforms.GaussianBlur(101, sigma=(10, 20))
    background = GaussianBlur(image)
    unsqueezed = saliency.view(1,1,224,224).expand_as(image).detach()
    blurred = image.detach().clone().view(1,3,224,224)
    blurred = image.detach().clone().view(1,3,224,224)
    
    if mode == 'positive':
        blurred[unsqueezed<0] = background[unsqueezed<0]
    else:
        blurred[unsqueezed>0] = background[unsqueezed>0]
    return blurred

def equal_blur(x, saliency_1, saliency_2, baseline = 'blur', mode = 'positive'):
    
    if mode == 'positive':
        n_mask = min((saliency_1>0).float().sum(), (saliency_2>0).float().sum())
    elif mode == 'negative':
        n_mask = max((saliency_1<0).float().sum(), (saliency_2<0).float().sum())
    else:
        print('Wrong Mode!')
        
    sal1 = saliency_1.view(-1, 224*224).argsort().argsort()
    sal2 = saliency_2.view(-1, 224*224).argsort().argsort()
    
    mask1 = torch.ones_like(sal1).to(x.device)
    mask2 = torch.ones_like(sal1).to(x.device)
    
    
    if mode == 'positive':
        mask1[sal1<n_mask] = 0
        mask2[sal2<n_mask] = 0
    elif mode == 'negative':
        mask1[sal1>224*224 - n_mask] = 0
        mask2[sal2>224*224 - n_mask] = 0
    else:
        print('Wrong Mode!')
        
    
    if baseline == 'blur':
        GaussianBlur = transforms.GaussianBlur(101, sigma=(10, 20))
        blur = GaussianBlur(x)
    elif baseline == 'zero':
        blur = torch.zeros(x.shape).to(x.device)
    elif baseline == 'mean':
        blur = x.mean(dim = [2,3]).view(1,3,1,1).expand_as(x)
    else:
        print('No Baseline!')
    
    mask1 = mask1.view(1,1,224,224).expand(x.shape) 
    mask2 = mask2.view(1,1,224,224).expand(x.shape)   
    
    masked_image1 = x.detach().clone()
    masked_image2 = x.detach().clone()
    
    masked_image1 = masked_image1*mask1 + (1-mask1)*blur
    masked_image2 = masked_image2*mask2 + (1-mask2)*blur
    return masked_image1, masked_image2

    

def get_samples(testset,model,threshold=0.3,max_no_samples=None,no_targets=2):
   
    model.eval()
    samples = []
    with torch.no_grad():
        for n in range(len(testset)):
            x = testset[n][0].view(1,3,224,224).to(next(model.parameters()).device)
            prob = torch.softmax(model(x),dim =1)[0]
            if prob.sort()[0][-no_targets]>threshold:
                sample = [n]
                print(f"{n}-th/{len(testset)},\t")
                for i in range(1, no_targets+1):
                    print(f"{i} as {prob.sort()[1][-i].item()}",
                          ": {:.4f}".format(prob.sort()[0][-i].item()))
                    sample.append(prob.sort()[1][-i].item())
                
                samples.append(sample)
                
                if max_no_samples and len(samples) >= max_no_samples:
                    break
    return samples
