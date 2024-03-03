import sys, os
import pickle
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from config import config
from models.transformer.decoder_vision import SkeletonTransformer
# from models.transformer.visionencoder_torchdecoder import SkeletonTransformer
# from models.transformer.perceiverencoder_torchdecoder import SkeletonTransformer
from data.interact_video import INTERACTDataset
from torch.autograd import Variable
import torch.optim as optim
import math
from torch.optim.lr_scheduler import LambdaLR
import wandb

shuffle = False
sampler = None


n_epochs = config.epochs
grad_clip = 1.0

n_characters = config.input_dim
BATCH_SIZE = config.batch_size
log_every = 1
save_every = 5*5
temperature = 0.0

train_dataset = INTERACTDataset(config, 'train', config.data_aug, exp_type=config.exp_type)
train_iterator = DataLoader(train_dataset, batch_size=config.batch_size,
                        num_workers=config.num_workers, drop_last=True,
                        sampler=sampler, shuffle=shuffle, pin_memory=True)

test_dataset = INTERACTDataset(config, 'test', config.data_aug, exp_type=config.exp_type)
test_iterator = DataLoader(test_dataset, batch_size=config.batch_size,
                        num_workers=config.num_workers, drop_last=True,
                        sampler=sampler, shuffle=shuffle, pin_memory=True)

val_iterator = train_iterator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# torch.cuda.set_device(0)


if config.wandb_activate == True:
    wandb.init(project="{}_corrected_transformer_baseline_rivanna".format(config.exp_type), entity="crl_rl2", dir="/scratch/msy9an/tro_log/")
    wandb.config.update(config)
    
    
def to_np(x):
    return  x.data.cpu().numpy()

def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
        x = x.squeeze(0).permute(1,0,2)

    return Variable(x)


def restore_pose(actual_pose, predicted_pose, transform=True):
    """
    Loads scaler from pickle and restores the poses
    """
    # actual_pose = actual_pose.permute(1,0,2); predicted_pose = predicted_pose.permute(1,0,2)
    
    return actual_pose, predicted_pose


def euclidean(actual_pose, predicted_pose):
    euc = 0
    actual_pose = actual_pose.permute(1, 0, 2)
    predicted_pose = predicted_pose.permute(1, 0, 2)

    time_steps=  predicted_pose.shape[1]
    mse = (actual_pose-predicted_pose)**2
    #print (mse.shape)
    mse = mse*100*100
    mse = mse.reshape(-1, mse.size(-1))
    euc_tensor = torch.FloatTensor(mse.size(0), mse.size(-1)//3)
    for j in range(0,mse.size(-1), 3):

        euc_tensor[:,j//3] = (mse[:,j] + mse[:,j+1] + mse[:,j+2])
    euc_tensor = euc_tensor.reshape(-1, time_steps, euc_tensor.size(-1)).permute(1, 0, 2)

    unsqueezed_euc_tensor = torch.sqrt(euc_tensor)

    euc_tensor = torch.mean(unsqueezed_euc_tensor, [1,2], True).squeeze()    
    return euc_tensor,unsqueezed_euc_tensor#.mean()

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress / num_cycles))))

    return LambdaLR(optimizer, lr_lambda)

def forward(iterator, model, optimizer,temperature=0.5, eval_flag=True, scheduler=None):


    train_loss = 0; train_mse = 0; input_mse = 0; i =0; classifier_accuracy = 0; train_BCE = 0; final_t_mse= 0; D_loss = 0

    for i, (observation, target, video_motion_input) in enumerate(iterator):

        observation, target = to_var(observation), to_var(target)
        #reconstruction loss
        tgt_mask = torch.tril(torch.ones(config.motion.interact_input_length, config.motion.interact_input_length, device=device))
        if config.use_vision == True:
            # b, t, h, w, c = video_motion_input.shape
            # video_motion_input = video_motion_input.view(-1,h,w,c)
            # features = feature_extractor(video_motion_input.cuda())
            features = video_motion_input#.view(b,t, -1)#.permute(1,0,2)
            X_sample = model(observation.cuda(), target.cuda(), features.cuda(), tgt_mask=tgt_mask, inference=False)
        else:
            X_sample = model(observation.cuda(), target.cuda(), tgt_mask=tgt_mask, inference=False)

        transformed_target, transformed_decoded = restore_pose(target.cuda(), X_sample.cuda())
        transformed_input, transformed_input = restore_pose(observation, observation)
        # Compute the loss        
        
        loss = (X_sample.cuda()-target.cuda())**2
        loss = loss.mean()
        optimizer.zero_grad()        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()       
        scheduler.step()
        train_mse += (euclidean(transformed_target.cuda(), transformed_decoded.cuda()))[0]
        input_mse += (euclidean(transformed_target.cuda(), transformed_input[-1,:].repeat(transformed_target.shape[0],1,1).cuda()))[0]
        final_t_mse += ((transformed_target[:,-1] - transformed_decoded[:,-1].cuda())**2).mean()
        
        train_loss += loss
        train_BCE += loss
        
    return train_loss, train_mse, input_mse, train_BCE, final_t_mse


model = SkeletonTransformer(config.input_dim, config.d_model, config.num_layers, config.nhead, config.activation, config.learnable_embedding, config.dropout, config.use_vision, config.autoregressive)

# if gpu_count==2:
#     model = nn.DataParallel(model, device_ids=[0, 1])
print (model)
model = model.cuda()
best_test_mse = 100000

#encode/decode optimizers
optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
# optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
scheduler = get_cosine_schedule_with_warmup(optimizer, config.warmup_steps, 2*config.epochs * len(train_iterator))

best_val_mse = float("Inf")
patience_counter = 0
for epoch in range(n_epochs):
    print (temperature)
    train_loss, train_mse, input_mse_train, train_BCE, finaltrain_mse = forward(train_iterator, model, optimizer,temperature, False, scheduler=scheduler)
    test_loss =0; test_mse=0; input_mse_test=0; test_BCE=0; diff_input = 0; perc_diff = 0; final_mse = 0; test_pck=0
    with torch.no_grad():
        for i, (observation, target, video_motion_input) in enumerate(test_iterator):

            observation, target = to_var(observation), to_var(target)
            #reconstruction loss
            tgt_mask = torch.tril(torch.ones(config.motion.interact_input_length, config.motion.interact_input_length, device=device))
            if config.use_vision == True:
                # b, t, h, w, c = video_motion_input.shape
                # video_motion_input = video_motion_input.view(-1,h,w,c)
                # features = feature_extractor(video_motion_input.cuda())
                # features = features.view(b,t, -1)#.permute(1,0,2)
                features = video_motion_input
                X_sample = model(observation.cuda(), observation.cuda(), features.cuda(), tgt_mask=tgt_mask)                
            else:
                X_sample = model(observation.cuda(), observation.cuda(), tgt_mask=tgt_mask)
            
            recon_loss = (X_sample.cuda()-target.cuda())**2
            recon_loss = recon_loss.mean()
            recon_loss = recon_loss/2
            test_loss += recon_loss*100*100
            
            transformed_target, transformed_decoded = restore_pose(target.cuda(), X_sample.cuda())
            transformed_input, transformed_input = restore_pose(observation.cuda(), observation.cuda())
            squeezed_tensor, unsqueezed_tensor_preds = euclidean(transformed_target.cuda(), transformed_decoded.cuda())
            test_mse += squeezed_tensor

            squeezed_tensor, unsqueezed_tensor_diff = euclidean(transformed_target.cuda(), transformed_input[-1].repeat(transformed_target.shape[0],1,1).cuda())     
            # squeezed_tensor += (euclidean(transformed_target.cuda(), transformed_input[-1,:].repeat(transformed_target.shape[0],1,1).cuda()))[0]

            diff_input += squeezed_tensor#euclidean(transformed_target, transformed_input[:,-1].repeat(transformed_target.shape[1],1,1).permute(1,0,2))[0]
            final_mse += (abs(transformed_decoded[:,-1].cuda() - transformed_target[:,-1].cuda())**2).mean()
        # if epoch >1:
        #     test_dataset.plot_test_skeletons(transformed_target.reshape(-1, transformed_decoded.shape[-1]).detach().cpu().numpy())


        if epoch % log_every == 0:
            temperature = 0.5
            print ("test loader ", len(test_iterator), "train loader ", len(train_iterator))
            print('{} train loss ={} train bce ={} (mse_target={}) (mse_input={}) ' .format(
                epoch, train_loss/len(train_iterator), train_BCE/len(train_iterator), train_mse, input_mse_train
            ))                        
            print('[{}] test_loss={} ( mse_target={}) avg = {}' .format(
                        epoch, test_loss/len(test_iterator), test_mse, test_mse.mean()/len(test_iterator)
                    ))
            
            print('[{}] test_pck={} ( diff={}) avg = {}'.format(
                epoch, test_pck/len(test_iterator), diff_input, diff_input.mean()/len(test_iterator)
            ))

            print('')

        if config.wandb_activate == True:
            wandb.log({'Loss/train': train_loss.mean()/len(train_iterator)}, step=epoch)
            wandb.log({'Loss/test': test_mse.mean()/len(test_iterator)}, step=epoch)
            
            for i in range(0,train_mse.shape[-1]):
                wandb.log({'ADE{}/test'.format(i): test_mse[i]}, step=epoch)
                wandb.log({'ADE{}/train'.format(i): train_mse[i]}, step=epoch)
                    
        if best_test_mse >= test_mse.mean():
            best_test_mse = test_mse.mean()
            print ("best value epoch ", epoch)
                
