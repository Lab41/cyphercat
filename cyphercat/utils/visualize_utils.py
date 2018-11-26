#!/usr/bin/python3
"""
Set of functions used to call a series of algorithms used to visualize the object localization of a pre-trained 
network in PyTorch.  The different algorithms are discussed in several papers, while the implementation is based, 
roughly, on work in the following repository (https://github.com/sar-gupta/weakly-supervised-localization-survey)
"""

import numpy as np
import PIL


import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable

def saliency_map_general(model, input, label, plot = False):
    """
    saliency_map_general: implementation to return the most general form of the saliency map, informing
    on the regions of interest that activate a specific label.
    Args:
        - model: (PyTorch) Trained model trying to understand 
        - input: Image to be classfied and understood, passed as a PyTorch tensor (C x W x H)
        - label: Class to identify the regions of interest
    return: numpy array with heatmap data
    """
    input  =  Variable(input.unsqueeze_(0),requires_grad = True)
    output = model.forward(input)
    model.zero_grad()

    output[0][label].backward()

    grads = input.grad.data.clamp(min=0)
    grads.squeeze_()
    grads.transpose_(0,1)
    grads.transpose_(1,2)
    grads = np.amax(grads.cpu().numpy(), axis=2)
    
    grads -= grads.min()
    grads /= grads.max()
    
    grads *= 255
    grads = grads.astype(int)
    
    return grads


def guided_saliency_map(model, input, label, plot = False):
    """
    guided_saliency_map: implementation to return a guided saliency map, informing
    on the regions of interest that activate a specific label.
    Args:
        - model: (PyTorch) Trained model trying to understand 
        - input: Image to be classfied and understood, passed as a PyTorch tensor (C x W x H)
        - label: Class to identify the regions of interest
    return: numpy array with heatmap data 
    """
    input = Variable(input.unsqueeze_(0), requires_grad=True)
    
    try:
        h = [0]*len(list(model.modules()))

        def hookfunc(module, gradInput, gradOutput):
            return tuple([(None if g is None else g.clamp(min=0)) for g in gradInput])

        for j, i in enumerate(list(model.modules())):
            h[j] = i.register_backward_hook(hookfunc)

        output = model.forward(input)
        model.zero_grad()


        output[0][label].backward()

        for i in range(len(list(model.modules()))):
            h[i].remove()
    except Exception as e:
        print(e)
        for i in range(len(list(model.modules()))):
            h[i].remove()
        
    grads = input.grad.data.clamp(min=0)
    grads.squeeze_()
    grads.transpose_(0,1)
    grads.transpose_(1,2)
    grads = np.amax(grads.cpu().numpy(), axis=2)
    
    grads -= grads.min()
    grads /= grads.max()
    
    grads *= 255
    grads = grads.astype(int)

    return grads

def gradcam(model, input, label, layer_name, plot=False):
    """
    gradcam: implementation to return a class activation map using the gradient of class score with each 
    of last conv layer filters.  Calculate weighted sum of gradients and filters to finally obtain a map 
    of size equal to size of filters.
    Args:
        - model: (PyTorch) Trained model trying to understand 
        - input: Image to be classfied and understood, passed as a PyTorch tensor (C x W x H)
        - label: Class to identify the regions of interest
        - layer_name: Name of the layer to target, should be the last CNN.
    return:
    PIL image with cativation map
    """
    imgs_shape = (input.shape[1], input.shape[2])
    rs = torchvision.transforms.Resize( imgs_shape  )

    #find the right layer
    last_conv = None
    for name, item in model._modules.items():
        if name == layer_name:
            last_conv = item

    if last_conv == None:
        print('Cant find target layer')
        return None

    pre_image = input
    global gcdata
    global gcgrads

    def bhook(module, gradInputs, gradOutputs):
        global gcgrads
        gcgrads = gradOutputs

    def fhook(module, input, output):
        global gcdata
        gcdata = output
        
    hb = last_conv.register_backward_hook(bhook)
    hf = last_conv.register_forward_hook(fhook)
        
    out = model(input.unsqueeze_(0))
    model.zero_grad()
    out[0, label].backward()
    
    hb.remove()
    hf.remove()
    
    gcdata = gcdata[0]
    gcgrads = gcgrads[0].squeeze()
    
    gcgrads = gcgrads.mean(dim=2, keepdim=True)
    gcgrads = gcgrads.mean(dim=1, keepdim=True)
    #
    gcdata = gcdata.mul(gcgrads)
    gcdata = gcdata.sum(dim=0, keepdim=True)
    gcdata = gcdata.clamp(min=0)
    
    gcdata -= gcdata.min()
    gcdata /= gcdata.max()

    toi = torchvision.transforms.ToPILImage()
    gcdata = np.array(rs(toi(gcdata.data.cpu())))

    input.squeeze()
    
    return gcdata

def guided_gradcam(model, input, label,layer_name,  plot = False):
    """
    guided_gradcam: returns a combination of a guided saliency map and class activation map. this combines 
    the sensitivity to different classes from gradcam toguether with the greater resolution of the
    saliency map.
    Args:
        - model: (PyTorch) Trained model trying to understand 
        - input: Image to be classfied and understood, passed as a PyTorch tensor (C x W x H)
        - label: Class to identify the regions of interest
        - layer_name: Name of the layer to target, should be the last CNN.
    return:
        PIL image with cativation map
    """
    gc = gradcam(model, input, label, layer_name, plot=False)

    guided = guided_saliency_map(model=model, input=input[0], label=label, plot=False)
    gc = gc * guided
    
    rs = torchvision.transforms.Resize((32,32))

    
    gc -= gc.min()
    gc = np.divide(gc, gc.max())
    gc *= 255
    gc = gc.astype(int)

    return gc

def smooth_guided_saliency_map(model, input, label, transform,x=10, percent_noise=10, plot = True):
    """
    smooth_guided_saliency_map: Implementation of guided saliency map accounting for the fact 
    small, local variations in the local derivatives lead to the apparent noise one sees. This implementation smooths
    these.
    Args:
        - model: (PyTorch) Trained model trying to understand 
        - input: Image to be classfied and understood, passed as a PyTorch tensor (C x W x H)
        - x: Number fo times to sample for the smoothing
        - percent_nois: Percentage of noise to be itroduced during sampling for smoothing
    return:
        PIL image with cativation map
    """
    tensor_input = input
    
    final_grad = torch.zeros(input.shape).cuda()
    final_grad = final_grad.unsqueeze(0)
        
    h = [0]*len(list(model.modules()))

    def hookfunc(module, gradInput, gradOutput):
        return tuple([(None if g is None else g.clamp(min=0)) for g in gradInput])

    for j, i in enumerate(list(model.modules())):
        h[j] = i.register_backward_hook(hookfunc)
        
    for i in range(x):
        temp_input = tensor_input
        noise = torch.from_numpy(np.random.normal(loc=0, scale=(percent_noise/100) * 
                                                  (tensor_input.max() - tensor_input.min()), 
                                                  size=temp_input.shape)).type(torch.cuda.FloatTensor)
        temp_input = (temp_input.cuda() + noise).cpu().numpy()
        temp_input = np.transpose(temp_input, (1,2,0) )
        temp_input = PIL.Image.fromarray(temp_input.astype(np.uint8))
        temp_input = Variable(transform(temp_input).unsqueeze(0).cuda(), requires_grad=True)

        output = model.forward(temp_input)
        model.zero_grad()
        output[0][label].backward()
        final_grad += temp_input.grad.data
        
    for i in range(len(list(model.modules()))):
        h[i].remove()
    
    grads = final_grad/x
    grads = grads.clamp(min=0)
    grads.squeeze_()
    grads.transpose_(0,1)
    grads.transpose_(1,2)
    grads = np.amax(grads.cpu().numpy(), axis=2)
    
    grads -= grads.min()
    grads /= grads.max()
    
    grads *= 255
    grads = grads.astype(int)

    return grads

def smooth_guided_gradcam(model, input, label, transform, layer_name, plot = False ):
    guided = smooth_guided_saliency_map(model, input, label,transform = transform,  plot = False)
    gc = gradcam(model, input, label, layer_name = layer_name,  plot=False)
    gc = gc * guided
    
    rs = torchvision.transforms.Resize((32,32))

    
    gc -= gc.min()
    gc = np.divide(gc, gc.max())
    gc *= 255
    gc = gc.astype(int)

    return gc
