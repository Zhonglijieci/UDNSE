import torch
torch.manual_seed(11)
from load_model import load_model
from units import load_image, split_image, binary_to_text, decode

import argparse
parser = argparse.ArgumentParser()

# path
parser.add_argument("--key_path", type=str, default="data/key.png", help='Key path')
parser.add_argument("--decryption_path", type=str, default="result/result.png", help='Encrypt image path')
#settings
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--transparency", type=float, default=0.5, help='Key multiplication coefficient')
parser.add_argument("--m_len", type=int, default=150, help='Code length')
parser.add_argument("--k", type=int, default=2, help='The number of bits encoded in 1-bit code')
parser.add_argument("--t", type=int, default=1, help='Number of split carrier images')
parser.add_argument("--models", type=str, default='resnet50', help='Model selection')
args = parser.parse_args()

# load_image
im_tensor = load_image(args.decryption_path, args.device)
key_tensor = torch.squeeze(load_image(args.key_path, args.device).detach())

# load model
model = load_model(args.models)
model = model.to(args.device)
model.eval()

# split image
m_image = split_image(im_tensor, args.t)
m = ''

# decryption
for i in range(args.t * args.t):
    m_tensor = m_image[i] + args.transparency * key_tensor
    im_tensor = torch.unsqueeze(m_tensor, 0)
    output = model(im_tensor)
    code = decode(output, args.m_len, args.k)
    m += code

print(m)
message = binary_to_text(m)
    
print("Message:", message)
