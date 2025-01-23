import torch
from loss_function import cal_loss1, show_false, cal_loss2
import torch.nn as nn
from PIL import Image
import numpy as np
from load_model import load_VGG

def encryption(q, im_tensor, key_tensor, m, model, opt):
    '''
    For the carrier image, in each iteration, we record the difference between 
    the decoding result in a specific model and the plain text after 
    stacking the encryption key as the first loss. 
    
    At the same time, in order to maintain the quality and style of the image, 
    we also retain the initial carrier image for easy calculation of the degree of 
    change in the carrier image during the iteration process to measure the 
    magnitude of the change in image quality.

    Finally, when the difference between the decoding result and the plain 
    text is 0, exit the loop and store the carrier image.

    '''

    # initialization
    im_tensor_adv = im_tensor + opt.transparency * key_tensor
    im_tensor_adv = im_tensor_adv.requires_grad_(True) 
    output= model(im_tensor_adv)
    im_tensor_old = im_tensor
    loss = cal_loss1(output, m, opt.m_len, opt.k)
    velocity = torch.zeros_like(im_tensor_adv, requires_grad=False)   

    vgg_net = load_VGG()
    vgg_net = vgg_net.to(opt.device)
    vgg_net.eval()

    # start
    cnt = 0
    success = False
    while success == False:
        cnt += 1
        
        loss.backward()
        velocity = opt.momentum * velocity + opt.lr * im_tensor_adv.grad 
        with torch.no_grad():
            im_tensor -= velocity 
        im_tensor = torch.clamp(im_tensor, 0, 1)

        uint8_tensor = (im_tensor.mul(255)).byte() 
        uint8_tensor1 = uint8_tensor.cpu().numpy()
        uint8_tensor1 = torch.from_numpy(uint8_tensor1).to(torch.float32)
        fp32_tensor_recovered = uint8_tensor1.float().div(255)
        fp32_tensor_recovered = fp32_tensor_recovered.to(opt.device)

        model.zero_grad()
        vgg_net.zero_grad()
        im_tensor_adv = fp32_tensor_recovered + opt.transparency * key_tensor
        im_tensor_adv = im_tensor_adv.detach() 
        im_tensor_adv = im_tensor_adv.requires_grad_(True)

        # calculate losses
        output_vgg_1 = vgg_net(im_tensor_old)
        output_vgg_2 = vgg_net(fp32_tensor_recovered)
        output= model(im_tensor_adv)
        output_old= model(im_tensor_old)

        loss1 = cal_loss1(output, m, opt.m_len, opt.k)
        loss2 = cal_loss2(output_vgg_1, output_vgg_2)
        loss3_function = nn.MSELoss()
        loss3 = loss3_function(output, output_old)

        s2 = 1000
        s3 = 0
        loss = loss1 + s2 * (loss2) + s3 * (loss3)

        # calculate error rate
        f_count, count0, count1, count_t, all_t = show_false(output, m, opt.m_len, opt.k)
        
        if cnt%100 ==0:
            print("Step: ", cnt, "   Number of errors: ", f_count, "/", opt.m_len)

        if f_count == 0:
            opt.total_cnt = opt.total_cnt + cnt
            im_tensor_1 = torch.squeeze(uint8_tensor.detach())
            im_tensor_1 = im_tensor_1.cpu().numpy()
            im_array = Image.fromarray(np.transpose(im_tensor_1, (1, 2, 0)))  
            im_array.save(f'output/'+str(q)+'.png', lossless=True)
            break