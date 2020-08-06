from datasets import get_imgs
from captioning_utils import get_hypothesis_greedy
from torch.autograd import Variable
import torchvision.transforms as transforms
import os
import torch
import json
import pickle
from tqdm import tqdm
from models import Encoder, DecoderWithAttention

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors

#################### Load the Imperfect Image->Text captioning model #######################
#model_location = '/data2/adsue/caption_dataset/pretrained/BEST_XE_checkpoint_7_coco_5_cap_per_img_5_min_word_freq.pth.tar'

attention_dim = 512
emb_dim = 512
decoder_dim = 512
dropout = 0.5
data_name =  'coco_5_cap_per_img_5_min_word_freq'

# Read word map
word_map_file = os.path.join('/scratch/scratch2/adsue/caption_dataset', 'WORDMAP_' + data_name + '.json')
with open(word_map_file, 'r') as j:
    word_map = json.load(j)
        
decoder = DecoderWithAttention(attention_dim=attention_dim,
                                       embed_dim=emb_dim,
                                       decoder_dim=decoder_dim,
                                       vocab_size=len(word_map),
                                       dropout=dropout)

decoder.load_state_dict(torch.load('/scratch/scratch2/adsue/pretrained/decoder_dict.pkl'))
decoder = decoder.to(device)
decoder.eval()

encoder = Encoder()
encoder.load_state_dict(torch.load('/scratch/scratch2/adsue/pretrained/encoder_dict.pkl'))
encoder = encoder.to(device)
encoder.eval()
##########################################################################################################################

imsize = 256
image_transform = transforms.Compose([
        transforms.Scale(int(imsize * 76 / 64)),
        transforms.RandomCrop(imsize)])


norm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])])

folder = '/scratch/scratch2/adsue/caption_dataset/train2014/' 
captions = {}
for file in tqdm(os.listdir(folder),desc='images'):
    #img_name = '%s/images/%s.jpg' % (data_dir, key)
    key = file.split('.')[0]
        
    imgs= get_imgs(os.path.join(folder,file), None,
                   None, image_transform, normalize=norm, single_level=True)
        

    imgs = Variable(torch.stack(imgs,0)).cuda()
               
    text = get_hypothesis_greedy(encoder(imgs), decoder)
    
    captions[key]=text

with open('train_imperfect_captions.pkl','wb') as f:
    pickle.dump(captions,f)
