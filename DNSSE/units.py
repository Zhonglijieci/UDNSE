import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

def load_image(path, device):

    preprocess = transforms.Compose([transforms.ToTensor()])
    I = Image.open(path).convert("RGB")
    im_tensor = preprocess(I)
    im_tensor = torch.unsqueeze(im_tensor, 0)
    im_tensor = im_tensor.to(device)
    return im_tensor

def split_image(input_tensor, m=1):
    
    input_tensor = torch.squeeze(input_tensor.detach())
    num_splits = m
    split_size = input_tensor.size(1) // num_splits
    split_tensors = []
    
    for i in range(num_splits):
        for j in range(num_splits):
            split = input_tensor[:, i * split_size: (i + 1) * split_size, j * split_size: (j + 1) * split_size]
            split_tensors.append(split)

    result_tensor = torch.stack(split_tensors)

    return result_tensor

def save_image(t, device, split_size, path):

    split_tensors = []
    for i in range(t*t):
        I = Image.open('output/'+str(i)+'.png').convert("RGB")
        im_tensor = transforms.ToTensor()(I)
        im_tensor = torch.unsqueeze(im_tensor,0)
        im_tensor = im_tensor.to(device)
        split_tensors.append(im_tensor)

    result_tensor = torch.stack(split_tensors)

    split_size = split_size
    restored_tensor = torch.zeros(3, t * split_size, t * split_size)

    for i in range(t):
        for j in range(t):
            restored_tensor[:, i * split_size: (i + 1) * split_size, j * split_size: (j + 1) * split_size] = result_tensor[i*t + j]

    restored_tensor = (restored_tensor.mul(255)).byte() 

    im_tensor_1 = torch.squeeze(restored_tensor.detach())
    im_tensor_1 = im_tensor_1.cpu().numpy()
    im_array = Image.fromarray(np.transpose(im_tensor_1, (1, 2, 0)))  
    im_array.save(path)

def text_to_binary(text):

    binary_representation = ''
    for char in text:
        ascii_value = ord(char)
        binary_representation += format(ascii_value, '08b')

    return binary_representation

def binary_to_text(binary_string):

    text = ''
    for i in range(0, len(binary_string), 8):
        binary_chunk = binary_string[i:i+8]
        ascii_value = int(binary_chunk, 2)
        text += chr(ascii_value)

    return text

def enlongcode(code, num):

    length = len(code)
    t = num - length
    code += '0' * t

    return code

def decode(outputs, length, k=2):

    code = ''
    for i in range(length):
        t = 0
        
        for j in range(k):
            t += outputs[0][i + j * length]
        t = t.item()
        if(t>0.8):
            code += '1'
        elif(t<0.8):
            code += '0'

    return code