from __future__ import print_function
from six.moves import range
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from PIL import Image

from miscc.config import cfg
from miscc.utils import mkdir_p
from miscc.utils import build_super_images, build_super_images2
from miscc.utils import weights_init, load_params, copy_G_params
from model import G_DCGAN, G_NET
from datasets import prepare_data, sort_by_keys
from model import RNN_ENCODER, CNN_ENCODER
from VGGFeatureLoss import VGGNet


from miscc.losses import words_loss
from miscc.losses import discriminator_loss, generator_loss, KL_loss
from miscc.losses import negative_ddva

from copy import deepcopy

import os
import time
import numpy as np
import sys

secondary_device = torch.device("cuda:"+str(cfg.secondary_GPU_ID))

# ################# Text to image task############################ #
class condGANTrainer(object):
    def __init__(self, output_dir, data_loader, n_words, ixtoword):
        if cfg.TRAIN.FLAG:
            self.model_dir = os.path.join(output_dir, 'Model')
            self.image_dir = os.path.join(output_dir, 'Image')
            mkdir_p(self.model_dir)
            mkdir_p(self.image_dir)

        torch.cuda.set_device(cfg.GPU_ID)
        cudnn.benchmark = True

        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.max_epoch = cfg.TRAIN.MAX_EPOCH
        self.snapshot_interval = cfg.TRAIN.SNAPSHOT_INTERVAL

        self.n_words = n_words
        self.ixtoword = ixtoword
        self.data_loader = data_loader
        self.num_batches = len(self.data_loader)

    def build_models(self):

        if cfg.TRAIN.NET_E == '':
            print('Error: no pretrained text-image encoders')
            return
        
        # vgg16 network
        style_loss = VGGNet()
        
        
        for p in style_loss.parameters():
            p.requires_grad = False

        print("Load the style loss model")
        style_loss.eval()

        image_encoder = CNN_ENCODER(cfg.TEXT.EMBEDDING_DIM)
        img_encoder_path = cfg.TRAIN.NET_E.replace('text_encoder', 'image_encoder')
        state_dict = \
            torch.load(img_encoder_path, map_location=lambda storage, loc: storage)
        image_encoder.load_state_dict(state_dict)
        for p in image_encoder.parameters():
            p.requires_grad = False
        print('Load image encoder from:', img_encoder_path)
        image_encoder.eval()

        text_encoder = \
            RNN_ENCODER(self.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
        state_dict = \
            torch.load(cfg.TRAIN.NET_E,
                       map_location=lambda storage, loc: storage)
        text_encoder.load_state_dict(state_dict)
        for p in text_encoder.parameters():
            p.requires_grad = False
        print('Load text encoder from:', cfg.TRAIN.NET_E)
        text_encoder.eval()

        netsD = []
        if cfg.GAN.B_DCGAN:
            if cfg.TREE.BRANCH_NUM ==1:
                from model import D_NET64 as D_NET
            elif cfg.TREE.BRANCH_NUM == 2:
                from model import D_NET128 as D_NET
            else:  # cfg.TREE.BRANCH_NUM == 3:
                from model import D_NET256 as D_NET
            netG = G_DCGAN()
            netsD = [D_NET(b_jcu=False)]
        else:
            from model import D_NET64, D_NET128, D_NET256
            netG = G_NET()
            
            if cfg.TREE.BRANCH_NUM > 0:
                netsD.append(D_NET64())
            if cfg.TREE.BRANCH_NUM > 1:
                netsD.append(D_NET128())
            if cfg.TREE.BRANCH_NUM > 2:
                netsD.append(D_NET256())
        netG.apply(weights_init)
        
        for i in range(len(netsD)):
            netsD[i].apply(weights_init)
        print('# of netsD', len(netsD))
        #
        epoch = 0
        if cfg.TRAIN.NET_G != '':
            state_dict = \
                torch.load(cfg.TRAIN.NET_G, map_location=lambda storage, loc: storage)
            netG.load_state_dict(state_dict)
            
            print('Load G from: ', cfg.TRAIN.NET_G)
            istart = cfg.TRAIN.NET_G.rfind('_') + 1
            iend = cfg.TRAIN.NET_G.rfind('.')
            epoch = cfg.TRAIN.NET_G[istart:iend]
            epoch = int(epoch) + 1
            if cfg.TRAIN.B_NET_D:
                Gname = cfg.TRAIN.NET_G
                for i in range(len(netsD)):
                    s_tmp = Gname[:Gname.rfind('/')]
                    Dname = '%s/netD%d.pth' % (s_tmp, i)
                    print('Load D from: ', Dname)
                    state_dict = \
                        torch.load(Dname, map_location=lambda storage, loc: storage)
                    netsD[i].load_state_dict(state_dict)
                    
        # Create a target network.
        target_netG = deepcopy(netG)

        if cfg.CUDA:
            text_encoder = text_encoder.cuda()
            image_encoder = image_encoder.cuda()
            style_loss = style_loss.cuda()     
            
            # The target network is stored on the scondary GPU.---------------------------------
            target_netG.cuda(secondary_device) 
            target_netG.ca_net.device = secondary_device
            #-----------------------------------------------------------------------------------
            
            netG.cuda()
            for i in range(len(netsD)):
                netsD[i]=  netsD[i].cuda()

                
        # Disable training in the target network:
        for p in target_netG.parameters():
            p.requires_grad = False            
                
        return [text_encoder, image_encoder, netG, target_netG, netsD, epoch, style_loss]

    def define_optimizers(self, netG, netsD):
        optimizersD = []
        num_Ds = len(netsD)
        for i in range(num_Ds):
            opt = optim.Adam(netsD[i].parameters(),
                             lr=cfg.TRAIN.DISCRIMINATOR_LR,
                             betas=(0.5, 0.999))
            optimizersD.append(opt)

        optimizerG = optim.Adam(netG.parameters(),
                                lr=cfg.TRAIN.GENERATOR_LR,
                                betas=(0.5, 0.999))

        return optimizerG, optimizersD

    def prepare_labels(self):
        batch_size = self.batch_size
        real_labels = Variable(torch.FloatTensor(batch_size).fill_(1))
        fake_labels = Variable(torch.FloatTensor(batch_size).fill_(0))
        match_labels = Variable(torch.LongTensor(range(batch_size)))
        if cfg.CUDA:
            real_labels = real_labels.cuda()
            fake_labels = fake_labels.cuda()
            match_labels = match_labels.cuda()

        return real_labels, fake_labels, match_labels

    def save_model(self, netG, avg_param_G, netsD, epoch):
        save_dir = '/scratch/scratch2/adsue/checkpoints2'
        backup_para = copy_G_params(netG)
        load_params(netG, avg_param_G)
        torch.save(netG.state_dict(),
            '%s/netG_epoch_%d.pth' % (save_dir, epoch))
        load_params(netG, backup_para)
        #
        for i in range(len(netsD)):
            netD = netsD[i]
            torch.save(netD.state_dict(),
                '%s/netD%d.pth' % (save_dir, i))
        print('Save G/Ds models.')

    def set_requires_grad_value(self, models_list, brequires):
        for i in range(len(models_list)):
            for p in models_list[i].parameters():
                p.requires_grad = brequires

    def save_img_results(self, netG, noise, sent_emb, words_embs, mask,
                         image_encoder, captions, cap_lens,
                         gen_iterations, name='current'):
        # Save images
        fake_imgs, attention_maps, _, _ = netG(noise, sent_emb, words_embs, mask)
        for i in range(len(attention_maps)):
            if len(fake_imgs) > 1:
                img = fake_imgs[i + 1].detach().cpu()
                lr_img = fake_imgs[i].detach().cpu()
            else:
                img = fake_imgs[0].detach().cpu()
                lr_img = None
            attn_maps = attention_maps[i]
            att_sze = attn_maps.size(2)
            img_set, _ = \
                build_super_images(img, captions, self.ixtoword,
                                   attn_maps, att_sze, lr_imgs=lr_img)
            if img_set is not None:
                im = Image.fromarray(img_set)
                fullpath = '%s/G_%s_%d_%d.png'\
                    % (self.image_dir, name, gen_iterations, i)
                im.save(fullpath)

        i = -1
        img = fake_imgs[i].detach()
        region_features, _ = image_encoder(img)
        att_sze = region_features.size(2)
        _, _, att_maps = words_loss(region_features.detach(),
                                    words_embs.detach(),
                                    None, cap_lens,
                                    None, self.batch_size)
        img_set, _ = \
            build_super_images(fake_imgs[i].detach().cpu(),
                               captions, self.ixtoword, att_maps, att_sze)
        if img_set is not None:
            im = Image.fromarray(img_set)
            fullpath = '%s/D_%s_%d.png'\
                % (self.image_dir, name, gen_iterations)
            im.save(fullpath)

    def train(self):
        text_encoder, image_encoder, netG, target_netG, netsD, start_epoch, style_loss = self.build_models()
        avg_param_G = copy_G_params(netG)
        optimizerG, optimizersD = self.define_optimizers(netG, netsD)
        real_labels, fake_labels, match_labels = self.prepare_labels()

        batch_size = self.batch_size
        nz = cfg.GAN.Z_DIM
        noise = Variable(torch.FloatTensor(batch_size, nz))
        fixed_noise = Variable(torch.FloatTensor(batch_size, nz).normal_(0, 1))
        if cfg.CUDA:
            noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

        gen_iterations = 0

        for epoch in range(start_epoch, self.max_epoch):
            start_t = time.time()

            data_iter = iter(self.data_loader)
            step = 0
            while step < self.num_batches:

                data = data_iter.next()
                
                captions, cap_lens, imperfect_captions, imperfect_cap_lens, misc = data
                
                # Generate images for human-text ----------------------------------------------------------------
                data_human = [captions, cap_lens, misc]
                
                imgs, captions, cap_lens, class_ids, keys, wrong_caps, \
                                wrong_caps_len, wrong_cls_id = prepare_data(data_human)

                hidden = text_encoder.init_hidden(batch_size)
                words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
                words_embs, sent_emb = words_embs.detach(), sent_emb.detach()

                # wrong word and sentence embeddings
                w_words_embs, w_sent_emb = text_encoder(wrong_caps, wrong_caps_len, hidden)
                w_words_embs, w_sent_emb = w_words_embs.detach(), w_sent_emb.detach()

                mask = (captions == 0)
                num_words = words_embs.size(2)
                if mask.size(1) > num_words:
                    mask = mask[:, :num_words]

                noise.data.normal_(0, 1)
                fake_imgs, _, mu, logvar = netG(noise, sent_emb, words_embs, mask)

                # Generate images for imperfect caption-text-------------------------------------------------------

                data_imperfect = [imperfect_captions, imperfect_cap_lens, misc]
                    
                imgs, imperfect_captions, imperfect_cap_lens, i_class_ids, imperfect_keys, i_wrong_caps,\
                            i_wrong_caps_len, i_wrong_cls_id = prepare_data(data_imperfect)
                    
                i_hidden = text_encoder.init_hidden(batch_size)
                i_words_embs, i_sent_emb = text_encoder(imperfect_captions, imperfect_cap_lens, i_hidden)
                i_words_embs, i_sent_emb = i_words_embs.detach(), i_sent_emb.detach()
                i_mask = (imperfect_captions == 0)
                i_num_words = i_words_embs.size(2)
                
                if i_mask.size(1) > i_num_words:
                    i_mask = i_mask[:, :i_num_words]

                # Move tensors to the secondary device.
                noise  = noise.to(secondary_device) # IMPORTANT! We are reusing the same noise.
                i_sent_emb = i_sent_emb.to(secondary_device)
                i_words_embs = i_words_embs.to(secondary_device)
                i_mask = i_mask.to(secondary_device)
                                
                # Generate images.
                imperfect_fake_imgs, _, _, _ = target_netG(noise, i_sent_emb, i_words_embs, i_mask)   
                    
                # Sort the results by keys to align ------------------------------------------------------------------------
                bag = [sent_emb, real_labels, fake_labels, words_embs, class_ids, w_words_embs, wrong_caps_len, wrong_cls_id]
                
                keys, captions, cap_lens, fake_imgs, _, sorted_bag = sort_by_keys(keys, captions, cap_lens, fake_imgs,\
                                                                                  None, bag)
                    
                sent_emb, real_labels, fake_labels, words_embs, class_ids, w_words_embs, wrong_caps_len, wrong_cls_id = \
                            sorted_bag
                 
                imperfect_keys, imperfect_captions, imperfect_cap_lens, imperfect_fake_imgs, imgs, _ = \
                            sort_by_keys(imperfect_keys, imperfect_captions, imperfect_cap_lens, imperfect_fake_imgs, imgs,None)
                    
                #-----------------------------------------------------------------------------------------------------------
                
                                
                errD_total = 0
                D_logs = ''
                for i in range(len(netsD)):
                    netsD[i].zero_grad()
                    errD = discriminator_loss(netsD[i], imgs[i], fake_imgs[i],
                                              sent_emb, real_labels, fake_labels,
                                              words_embs, cap_lens, image_encoder, class_ids, w_words_embs, 
                                              wrong_caps_len, wrong_cls_id)
                    # backward and update parameters
                    errD.backward(retain_graph=True)
                    optimizersD[i].step()
                    errD_total += errD
                    D_logs += 'errD%d: %.2f ' % (i, errD)

                step += 1
                gen_iterations += 1

                netG.zero_grad()
                errG_total, G_logs = \
                    generator_loss(netsD, image_encoder, fake_imgs, real_labels,
                                   words_embs, sent_emb, match_labels, cap_lens, class_ids, style_loss, imgs)
                kl_loss = KL_loss(mu, logvar)
                
                errG_total += kl_loss
                
                G_logs += 'kl_loss: %.2f ' % kl_loss
                
                
                # Shift device for the imgs and target_imgs.-----------------------------------------------------
                for i in range(len(imgs)):
                    imgs[i] = imgs[i].to(secondary_device)
                    fake_imgs[i] = fake_imgs[i].to(secondary_device)
                
                # Compute and add ddva loss ---------------------------------------------------------------------
                neg_ddva = negative_ddva(imperfect_fake_imgs, imgs, fake_imgs)
                neg_ddva *= 10. # Scale so that the ddva score is not overwhelmed by other losses.
                errG_total += neg_ddva.to(cfg.GPU_ID)
                G_logs += 'negative_ddva_loss: %.2f ' % neg_ddva
                #------------------------------------------------------------------------------------------------
                
                errG_total.backward()
                
                optimizerG.step()
                for p, avg_p in zip(netG.parameters(), avg_param_G):
                    avg_p.mul_(0.999).add_(0.001, p.data)

                if gen_iterations % 100 == 0:
                    print(D_logs + '\n' + G_logs)

                # Copy parameters to the target network.
                if gen_iterations % 20 == 0:
                    load_params(target_netG, copy_G_params(netG))

            end_t = time.time()

            print('''[%d/%d][%d]
                  Loss_D: %.2f Loss_G: %.2f neg_ddva: %.2f Time: %.2fs'''
                  % (epoch, self.max_epoch, self.num_batches,
                     errD_total, errG_total,
                     neg_ddva,
                     end_t - start_t))

            if epoch % cfg.TRAIN.SNAPSHOT_INTERVAL == 0: 
                self.save_model(netG, avg_param_G, netsD, epoch)

        self.save_model(netG, avg_param_G, netsD, self.max_epoch)

    def save_singleimages(self, images, filenames, save_dir,
                          split_dir, sentenceID=0):
        for i in range(images.size(0)):
            s_tmp = '%s/single_samples/%s/%s' %\
                (save_dir, split_dir, filenames[i])
            folder = s_tmp[:s_tmp.rfind('/')]
            if not os.path.isdir(folder):
                print('Make a new folder: ', folder)
                mkdir_p(folder)

            fullpath = '%s_%d.jpg' % (s_tmp, sentenceID)
            img = images[i].add(1).div(2).mul(255).clamp(0, 255).byte()
            ndarr = img.permute(1, 2, 0).data.cpu().numpy()
            im = Image.fromarray(ndarr)
            im.save(fullpath)

    def sampling(self, split_dir):
        if cfg.TRAIN.NET_G == '':
            print('Error: the path for models is not found!')
        else:
            if split_dir == 'test':
                split_dir = 'valid'
            if cfg.GAN.B_DCGAN:
                netG = G_DCGAN()
            else:
                netG = G_NET()
            netG.apply(weights_init)
            netG.cuda()
            netG.eval()
            #
            text_encoder = RNN_ENCODER(self.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
            state_dict = \
                torch.load(cfg.TRAIN.NET_E, map_location=lambda storage, loc: storage)
            text_encoder.load_state_dict(state_dict)
            print('Load text encoder from:', cfg.TRAIN.NET_E)
            text_encoder = text_encoder.cuda()
            text_encoder.eval()

            batch_size = self.batch_size
            nz = cfg.GAN.Z_DIM
            noise = Variable(torch.FloatTensor(batch_size, nz), volatile=True)
            noise = noise.cuda()

            model_dir = cfg.TRAIN.NET_G
            state_dict = \
                torch.load(model_dir, map_location=lambda storage, loc: storage)
            netG.load_state_dict(state_dict)
            print('Load G from: ', model_dir)

            # the path to save generated images
            s_tmp = model_dir[:model_dir.rfind('.pth')]
            save_dir = '%s/%s' % (s_tmp, split_dir)
            mkdir_p(save_dir)

            cnt = 0
            idx = 0 ###
            
            avg_ddva = 0
            for _ in range(1): 
                for step, data in enumerate(self.data_loader, 0):
                    cnt += batch_size
                    if step % 100 == 0:
                        print('step: ', step)
                        
                    captions, cap_lens, imperfect_captions, imperfect_cap_lens, misc = data
                               
                    
                    # Generate images for human-text ----------------------------------------------------------------
                    data_human = [captions, cap_lens, misc]
                    
                    imgs, captions, cap_lens, class_ids, keys, wrong_caps,\
                                wrong_caps_len, wrong_cls_id= prepare_data(data_human)
                    
                    hidden = text_encoder.init_hidden(batch_size)
                    words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
                    words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
                    mask = (captions == 0)
                    num_words = words_embs.size(2)
                    if mask.size(1) > num_words:
                        mask = mask[:, :num_words]

                    noise.data.normal_(0, 1)
                    fake_imgs, _, _, _ = netG(noise, sent_emb, words_embs, mask)
                    
                    # Generate images for imperfect caption-text-------------------------------------------------------
                    data_imperfect = [imperfect_captions, imperfect_cap_lens, misc]
                    
                    imgs, imperfect_captions, imperfect_cap_lens, class_ids, imperfect_keys, wrong_caps,\
                                wrong_caps_len, wrong_cls_id = prepare_data(data_imperfect)
                    
                    hidden = text_encoder.init_hidden(batch_size)
                    words_embs, sent_emb = text_encoder(imperfect_captions, imperfect_cap_lens, hidden)
                    words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
                    mask = (imperfect_captions == 0)
                    num_words = words_embs.size(2)
                    if mask.size(1) > num_words:
                        mask = mask[:, :num_words]

                    noise.data.normal_(0, 1)
                    imperfect_fake_imgs, _, _, _ = netG(noise, sent_emb, words_embs, mask)
                    
                    # Sort the results by keys to align ----------------------------------------------------------------
                    keys, captions, cap_lens, fake_imgs, _, _= sort_by_keys(keys, captions, cap_lens, fake_imgs, None, None)
                    
                    imperfect_keys, imperfect_captions, imperfect_cap_lens, imperfect_fake_imgs, true_imgs, _ = \
                                sort_by_keys(imperfect_keys, imperfect_captions, imperfect_cap_lens, imperfect_fake_imgs,\
                                             imgs, None)
                    
                    # Shift device for the imgs, target_imgs and imperfect_imgs------------------------------------------------
                    for i in range(len(imgs)):
                        imgs[i] = imgs[i].to(secondary_device)
                        imperfect_fake_imgs[i] = imperfect_fake_imgs[i].to(secondary_device)
                        fake_imgs[i] = fake_imgs[i].to(secondary_device)
                    
                    for j in range(batch_size):
                        s_tmp = '%s/single' % (save_dir)
                        folder = s_tmp[:s_tmp.rfind('/')]
                        if not os.path.isdir(folder):
                            print('Make a new folder: ', folder)
                            mkdir_p(folder)
                        k = -1
                        im = fake_imgs[k][j].data.cpu().numpy()
                        im = (im + 1.0) * 127.5
                        im = im.astype(np.uint8)
                        im = np.transpose(im, (1, 2, 0))
                        
                        cap_im = imperfect_fake_imgs[k][j].data.cpu().numpy()
                        cap_im = (cap_im + 1.0) * 127.5
                        cap_im = cap_im.astype(np.uint8)
                        cap_im = np.transpose(cap_im, (1, 2, 0))
                        
                        # Uncomment to scale true image
                        true_im = true_imgs[k][j].data.cpu().numpy()
                        true_im = (true_im + 1.0) * 127.5
                        true_im = true_im.astype(np.uint8)
                        true_im = np.transpose(true_im, (1, 2, 0))
                        
                        # Uncomment to save images.
                        #true_im = Image.fromarray(true_im)
                        #fullpath = '%s_true_s%d.png' % (s_tmp, idx)
                        #true_im.save(fullpath) 
                        im = Image.fromarray(im)
                        fullpath = '%s_s%d.png' % (s_tmp, idx)
                        im.save(fullpath)                        
                        #cap_im = Image.fromarray(cap_im)
                        #fullpath = '%s_imperfect_s%d.png' % (s_tmp, idx)
                        idx = idx+1
                        #cap_im.save(fullpath)
                        
                        
                    neg_ddva = negative_ddva(imperfect_fake_imgs, imgs, fake_imgs, reduce='mean', final_only=True).data.cpu().numpy()
                    avg_ddva += neg_ddva*(-1) 
                  
                    #text_caps = [[self.ixtoword[word] for word in sent if word!=0] for sent in captions.tolist()]
                    
                    #imperfect_text_caps = [[self.ixtoword[word] for word in sent if word!=0] for sent in
                    #                       imperfect_captions.tolist()]


                    
                    print(step)
            avg_ddva = avg_ddva/(step+1)
            print('\n\nAvg_DDVA: ', avg_ddva)

    def gen_example(self, data_dic):
        if cfg.TRAIN.NET_G == '':
            print('Error: the path for models is not found!')
        else:
            text_encoder = \
                RNN_ENCODER(self.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
            state_dict = \
                torch.load(cfg.TRAIN.NET_E, map_location=lambda storage, loc: storage)
            text_encoder.load_state_dict(state_dict)
            print('Load text encoder from:', cfg.TRAIN.NET_E)
            text_encoder = text_encoder.cuda()
            text_encoder.eval()

            if cfg.GAN.B_DCGAN:
                netG = G_DCGAN()
            else:
                netG = G_NET()
            s_tmp = cfg.TRAIN.NET_G[:cfg.TRAIN.NET_G.rfind('.pth')]
            model_dir = cfg.TRAIN.NET_G
            state_dict = \
                torch.load(model_dir, map_location=lambda storage, loc: storage)
            netG.load_state_dict(state_dict)
            print('Load G from: ', model_dir)
            netG.cuda()
            netG.eval()
            for key in data_dic:
                save_dir = '%s/%s' % (s_tmp, key)
                mkdir_p(save_dir)
                captions, cap_lens, sorted_indices = data_dic[key]

                batch_size = captions.shape[0]
                nz = cfg.GAN.Z_DIM
                captions = Variable(torch.from_numpy(captions), volatile=True)
                cap_lens = Variable(torch.from_numpy(cap_lens), volatile=True)

                captions = captions.cuda()
                cap_lens = cap_lens.cuda()
                for i in range(1): 
                    noise = Variable(torch.FloatTensor(batch_size, nz), volatile=True)
                    noise = noise.cuda()

                    hidden = text_encoder.init_hidden(batch_size)
                    words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
                    mask = (captions == 0)

                    noise.data.normal_(0, 1)
                    fake_imgs, attention_maps, _, _ = netG(noise, sent_emb, words_embs, mask)

                    cap_lens_np = cap_lens.cpu().data.numpy()
                    for j in range(batch_size):
                        save_name = '%s/%d_s_%d' % (save_dir, i, sorted_indices[j])
                        for k in range(len(fake_imgs)):
                            im = fake_imgs[k][j].data.cpu().numpy()
                            im = (im + 1.0) * 127.5
                            im = im.astype(np.uint8)
                            im = np.transpose(im, (1, 2, 0))
                            im = Image.fromarray(im)
                            fullpath = '%s_g%d.png' % (save_name, k)
                            im.save(fullpath)

                        for k in range(len(attention_maps)):
                            if len(fake_imgs) > 1:
                                im = fake_imgs[k + 1].detach().cpu()
                            else:
                                im = fake_imgs[0].detach().cpu()
                            attn_maps = attention_maps[k]
                            att_sze = attn_maps.size(2)
                            img_set, sentences = \
                                build_super_images2(im[j].unsqueeze(0),
                                                    captions[j].unsqueeze(0),
                                                    [cap_lens_np[j]], self.ixtoword,
                                                    [attn_maps[j]], att_sze)
                            if img_set is not None:
                                im = Image.fromarray(img_set)
                                fullpath = '%s_a%d.png' % (save_name, k)
                                im.save(fullpath)
