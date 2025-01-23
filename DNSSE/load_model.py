import torch
import torchvision.models as models
torch.manual_seed(11)


def load_model(model_name):
    if model_name == 'convnext':
        model = models.convnext_base(pretrained=True)   
    elif model_name == 'densenet':
        model = models.densenet121(pretrained=True)   
    elif model_name == 'efficientnet':
        model = models.efficientnet_b0(pretrained=True)   
    elif model_name == 'efficientnet_v2':
        model = models.efficientnet_v2_s(pretrained=True)   
    elif model_name == 'googlenet':
        model = models.googlenet(pretrained=True)   
    elif model_name == 'inception_v3':
        model = models.inception_v3(pretrained=True)   
    elif model_name == 'maxvit':
        model = models.maxvit_t(pretrained=True)   
    elif model_name == 'mnasnet':
        model = models.mnasnet1_0(pretrained=True)   
    elif model_name == 'mobilenet_v2':
        model = models.mobilenet_v2(pretrained=True)   
    elif model_name == 'mobilenet_v3':
        model = models.mobilenet_v3_large(pretrained=True)   
    elif model_name == 'regnet':
        model = models.regnet_y_8gf(pretrained=True)   
    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=True)   
    elif model_name == 'resnet18':
        model = models.resnet18(pretrained=True)   
    elif model_name == 'resnext':
        model = models.resnext101_32x8d(pretrained=True)   
    elif model_name == 'shufflenet':
        model = models.shufflenet_v2_x1_0(pretrained=True)   
    elif model_name == 'swin_t':
        model = models.swin_t(pretrained=True)   
    elif model_name == 'vgg16':
        model = models.vgg16(pretrained=True)   
    elif model_name == 'vit':
        model = models.vit_b_16(pretrained=True)   
    elif model_name == 'wide_resnet50':
        model = models.wide_resnet50_2(pretrained=True)   
    return model

def load_VGG():
    vgg_net = models.vgg16(pretrained=True)
    return vgg_net


