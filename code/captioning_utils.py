import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torch.nn.functional as F
import json

word_map_file = '/scratch/scratch2/adsue/caption_dataset/WORDMAP_coco_5_cap_per_img_5_min_word_freq.json'  # word map, ensure it's the same the data was encoded with and the model was trained with

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead


# Load word map (word2ix)
with open(word_map_file, 'r') as j:
    word_map = json.load(j)
rev_word_map = {v: k for k, v in word_map.items()}
vocab_size = len(word_map)


def get_hypothesis_greedy(encoder_out, decoder):
    
    batch_size = encoder_out.size(0)
    encoder_dim = encoder_out.size(-1)

    # Flatten image
    encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
    num_pixels = encoder_out.size(1)

    # Tensor to store the previous word at each step; now it is just <start>
    prev_words = torch.LongTensor([[word_map['<start>']]] * batch_size).to(device)  # (batch_size, 1)

    # Tensor to store generated sequences; now it is just <start>
    seqs = prev_words.tolist()  # (batch_size, 1)

    # Start decoding
    step = 0
    H, C = decoder.init_hidden_state(encoder_out)

    # Keep track of sum top scores for the REINFORCE algorithm.
    sum_top_scores = torch.zeros(batch_size).to(device)

    
    # Update the indexes of sequences that are incomplete.    
    incomplete_inds = [ind for ind, last_word in enumerate(prev_words.squeeze(1).tolist()) if last_word != word_map['<end>']]
    
    alphas = torch.zeros(batch_size, 100, num_pixels).to(device)
    
    # Note: Batch size changes as generated sequences come to an <end>. 
    while len(incomplete_inds)>0:

        embeddings = decoder.embedding(prev_words[incomplete_inds]).squeeze(1)  # (batch_size, embed_dim)

        awe, alpha = decoder.attention(encoder_out[incomplete_inds], H[incomplete_inds])  # (batch_size, encoder_dim), (batch_size, num_pixels)

        gate = decoder.sigmoid(decoder.f_beta(H[incomplete_inds]))  # gating scalar, (batch_size, encoder_dim)
        awe = gate * awe

        H[incomplete_inds], C[incomplete_inds] = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (H[incomplete_inds], C[incomplete_inds]))  # (batch_size, decoder_dim)

        scores = decoder.fc(H[incomplete_inds])  # (batch_size, vocab_size)
        scores = F.log_softmax(scores, dim=1) #(batch_size, vocab_size) 

        top_scores, top_words = scores.max(1)  # (batch_size) (batch_size)
        
        # Convert unrolled indices to actual indices of scores
        next_word_inds = top_words % vocab_size  # (batch_size)
        
        prev_words[incomplete_inds] = next_word_inds.unsqueeze(1) 
        
        # Add new words to sequences
        for (i, index) in enumerate(incomplete_inds):
            seqs[index].append(next_word_inds[i].item())
        
           
        
        incomplete_inds = [ind for ind, last_word in enumerate(prev_words.squeeze(1).tolist()) if last_word != word_map['<end>']]
        
        # Break if things have been going on too long
        if step > 50:
            break
        step += 1

        
    hypotheses = [[rev_word_map[w] for w in se if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}] for se in seqs]

    return hypotheses
