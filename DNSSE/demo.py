import torch
import time
torch.manual_seed(11)
from load_model import load_model
from encryption import encryption
from units import load_image, split_image, save_image, text_to_binary, enlongcode

import argparse
parser = argparse.ArgumentParser()
# path
parser.add_argument("--image_path", type=str, default="data/image.png", help='Image path')
parser.add_argument("--key_path", type=str, default="data/key.png", help='Key path')
parser.add_argument("--save_path", type=str, default="result/result.png", help='Save path')
# settings
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--transparency", type=float, default=0.5, help='Key multiplication coefficient')
parser.add_argument("--m_len", type=int, default=150, help='Code length')
parser.add_argument("--k", type=int, default=2, help='The number of bits encoded in 1-bit code')
parser.add_argument("--split_size", type=int, default=224, help='Split image size')
parser.add_argument("--t", type=int, default=1, help='Number of split carrier images')
parser.add_argument("--lr", type=float, default=1e-4, help='Learning rate')
parser.add_argument("--momentum", type=float, default=0.9, help='Momentum of SGD optimizer')
parser.add_argument("--models", type=str, default='resnet50', help='Model selection')
parser.add_argument("--total_cnt", type=int, default=0, help='Total steps')
# plain text
parser.add_argument("--text", type=str, default="Here is a message", help='Text input')
args = parser.parse_args()

def main(args):
    '''
    The main program of DNN based Novel Security Encryption Framework (DNSSE).

    Firstly, we encode the plain text and import the pre trained model 
    that needs to be used. You can also import your own model for steganalysis testing.

    Then, we process the carrier image, crop it, and perform encryption on the cropped 
    parts separately.

    After encryption is completed, reassemble the returned image.
    '''
    
    all_start_time = time.time()

    # encode
    code = text_to_binary(args.text)
    m = enlongcode(code, args.m_len)

    # load models
    model = load_model(args.models)
    model = model.to(args.device)
    model.eval()

    # load image
    im_tensor = load_image(args.image_path, args.device)
    key_tensor = torch.squeeze(load_image(args.key_path, args.device).detach())

    # split image
    t=args.t
    m_image = split_image(im_tensor, t).to(args.device)

    # encryption
    for i in range(t * t):
        message = m[i*args.m_len : (i+1)*args.m_len]
        input = m_image[i] 
        input = torch.unsqueeze(input, 0)
        encryption(i, input, key_tensor, message, model, args)
    
    print('Total steps: ', args.total_cnt)

    # save
    save_image(t, args.device, args.split_size, args.save_path)

    all_end_time = time.time()
    duration = all_end_time - all_start_time
    print(f"Total time:  {duration:.2f} s")

if __name__ == '__main__':
        main(args)
