import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.activation_functions import SwishGLU
from utils.rotary_embedding_torch import LearnablePositionalEmbedding

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # (max_seq_len, 1, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # print ("use ROPE Embeddings. Also check on an efficient way to load video files, and tgt mask")
        x = x + self.pe[:x.size(0), :]
        return x 

    def generate_positional_encoding(self, d_model, max_len):
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        return pe


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads  = num_heads
        self.head_dim = d_model // num_heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        
        self.fc_out = nn.Linear(d_model, d_model)
        
        
    def forward(self, query, key, value, attn_mask=None):
        mask=attn_mask

        query = query.permute(1, 0, 2)
        key = key.permute(1, 0, 2)
        value = value.permute(1, 0, 2)

        batch_size = query.shape[0]
        # Linearly project queries, keys and values
        query = self.q_linear(query)
        key = self.q_linear(key)
        value = self.q_linear(value)
        
        # Split the queries, keys and values into multiple heads
        query = query.view(batch_size, -1, self.num_heads, self.head_dim)
        key = key.view(batch_size, -1, self.num_heads, self.head_dim)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim)
        
        # Transpose to prepare for attention computation
        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)
        # Compute scaled dot-product attention
        scores = torch.matmul(query, key.permute(0, 1, 3, 2)) / self.head_dim
        if mask is not None:
            # mask = mask.reshape(batch_size, self.num_heads, mask.shape[1], mask.shape[2])
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = torch.nn.functional.softmax(scores, dim=-1)
        
        # Apply attention to values        
        output = torch.matmul(attention, value)
        
        # Reshape and concatenate heads
        output = output.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.d_model)
        output = output.permute(1, 0, 2)
        # Linearly project the concatenated heads
        output = self.fc_out(output)
        
        return output, scores
        

class PositionwiseFeedforward(nn.Module):
    def __init__(self, d_model, d_ff, activation):
        super(PositionwiseFeedforward, self).__init__()
        self.activation = activation
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        if self.activation == 'swishglu':
            self.actvn_func = SwishGLU(d_ff)
        else: 
            self.actvn_func = nn.ReLU()
        
    def forward(self, x):
        x = self.actvn_func(self.fc1(x))
        x = self.fc2(x)
        
        return x
        
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, activation, dropout):
        super(TransformerEncoderLayer, self).__init__()
        
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.feedforward = PositionwiseFeedforward(d_model, d_ff, activation)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        
    def forward(self, x, mask=None):
        # Multi-head self-attention; normalize before attention
        x_vel, x_acc = self.extract_velocity(x)
        
        x = self.norm1(x)
        x_vel = self.norm1(x_vel)
        x_acc = self.norm1(x_acc)

        attention_output = self.self_attention(x, x, x, mask)[0]
        x = x + self.dropout(attention_output)
        
        attention_output = self.self_attention(x, x_vel, x_vel, mask)[0]
        x_vel = x_vel + self.dropout(attention_output)

        attention_output = self.self_attention(x, x_acc, x_acc, mask)[0]
        x_acc = x_acc + self.dropout(attention_output)
        x = x + x_vel + x_acc
        # x = torch.cat((x, x_vel, x_acc), dim=1)
        # Position-wise feedforward layer
        x = self.norm2(x)
        ff_output = self.feedforward(x)
        x = x + self.dropout(ff_output)
        
        return x
        
    def extract_velocity(self, x_pos):
        x_vel = torch.zeros((x_pos.shape[0], x_pos.shape[1], x_pos.shape[2])).cuda()
        x_acc = torch.zeros((x_pos.shape[0], x_pos.shape[1], x_pos.shape[2])).cuda()
        x_vel[:, :-1, :] = x_pos[:, 1:,:] - x_pos[:, :-1, :]
        x_acc[:, :-1, :] = x_vel[:, 1:,:] - x_vel[:, :-1, :]

        return x_vel, x_acc

class TransformerEncoder(nn.Module)        :
    def __init__(self, num_layers, d_model, num_heads, d_ff, input_dim, activation, learnable_embedding, dropout, use_vision, max_len=50):
        super(TransformerEncoder, self).__init__()
        self.num_layers = num_layers
        self.learnable_embedding = learnable_embedding
        #if use_vision == "OldTrue":
        #    input_dim = input_dim + d_model
        self.layers = nn.ModuleList([TransformerEncoderLayer(d_model, num_heads, d_ff, activation, dropout) for _ in range(num_layers)])
        self.embedding = nn.Linear(input_dim, d_model)
        if self.learnable_embedding == True:
            self.pos_encoding = LearnablePositionalEmbedding(d_model, max_len)
        else:
            self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, vis_latent=None, mask=None):
        # Write the velocity and acceleration code here
        #if vis_latent != None:
        #    x = torch.cat((x, vis_latent), dim=-1)
        x = self.embedding(x)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        for layer in self.layers:
            x = layer(x, mask)
            
        return x
        

class TransformerDecoderLayer(nn.TransformerDecoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward):
        super(TransformerDecoderLayer, self).__init__(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward)
        self.activation = SwishGLU(dim_feedforward)
            


class GPTDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, max_seq_length, learnable_embedding):
        super(GPTDecoder, self).__init__()
        
        self.embedding = nn.Linear(vocab_size, d_model)
        dff = d_model//2
        # Positional encoding

        self.learnable_embedding = learnable_embedding

        if self.learnable_embedding == True:
            self.pos_encoding = LearnablePositionalEmbedding(d_model, max_seq_length)
        else:
            self.pos_encoding = PositionalEncoding(d_model, max_seq_length)

        self.transformer_decoder = nn.TransformerDecoder(
            TransformerDecoderLayer(d_model, nhead, dff),
            num_layers=num_layers
        )
        
        self.fc1 = nn.Linear(d_model, d_model*2)
        self.fc2 = nn.Linear(d_model*2, vocab_size)
        self.activation = SwishGLU(d_model*2)

        
    def forward(self, tgt, memory, tgt_mask=None):
        _input = tgt
        tgt = self.embedding(tgt)  
        tgt = self.pos_encoding(tgt)        
        output = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask) #, tgt_mask=tgt_mask)
        output = self.fc2(self.activation(self.fc1(output))) + _input
        return output 
    

class SkeletonTransformer(nn.Module):        
    def __init__(self, input_dim, d_model, num_layers, nhead, activation, learnable_embedding, dropout, use_vision, auto_regressive=False):
        super(SkeletonTransformer, self).__init__()
        self.learnable_embedding = learnable_embedding
        self.auto_regressive = auto_regressive
        self.use_vision = use_vision
        self.agent_num = 3
        self.encoder = TransformerEncoder(num_layers=num_layers, d_model=d_model, num_heads=nhead, d_ff=d_model, 
                       input_dim=input_dim//self.agent_num, activation=activation, learnable_embedding=self.learnable_embedding, 
                       dropout=dropout, use_vision =use_vision, max_len=25)
        self.agent_attention = nn.MultiheadAttention(d_model, nhead)
        self.agent_attention_dropout = nn.Dropout(dropout)
        self.agent_pe = LearnablePositionalEmbedding(d_model, max_seq_len=25)
        self.decoder = GPTDecoder(input_dim//3, d_model, nhead, num_layers, max_seq_length=25, learnable_embedding=self.learnable_embedding)


        self.vis_pe = LearnablePositionalEmbedding(d_model, max_seq_len=25) 
        self.vis_norm = nn.LayerNorm(768)
        self.vis_proj1 = nn.Linear(768, d_model*2)
        self.vis_proj2 = nn.Linear(d_model*2, d_model)
        self.vis_act = SwishGLU(d_model*2)
        
    def forward(self, src, tgt, vis_memory=None, tgt_mask=None, inference=True):
        # Encoder
        delimiter = src.shape[-1]//3
        agents = list()
        latents = list()
        agent_latent = list()
        decoder_output = list()

        for i in range(self.agent_num):
            agents.append(src[:, :, delimiter*i:(delimiter*(i+1))] )        
        
        for i in range(self.agent_num):
            latents.append(self.encoder(agents[i]))
        
        agent_latent.append(self.compute_attention(latents[0], latents[1], latents[2]))
        agent_latent.append(self.compute_attention(latents[1], latents[0], latents[2]))
        agent_latent.append(self.compute_attention(latents[2], latents[0], latents[1]))


        if vis_memory!=None:
            b, t, c = vis_memory.shape
            vis_latent = vis_memory.view(b, t, -1).permute(1,0,2)
            vis_latent = self.vis_norm(vis_latent)
            vis_latent = self.vis_act(self.vis_proj1(vis_latent.reshape(-1, vis_latent.shape[-1])))
            vis_latent = (self.vis_proj2(vis_latent)).reshape(t, b, -1)
            vis_latent = self.vis_pe(vis_latent)
            
            agent_latent[0] = torch.cat((agent_latent[0], vis_latent), dim=0)
            agent_latent[1] = torch.cat((agent_latent[1], vis_latent), dim=0)
            agent_latent[2] = torch.cat((agent_latent[2], vis_latent), dim=0)

        # Decoder Input for the first step
        decoder_input = torch.zeros(tgt.shape[1], tgt.shape[2]).cuda()
        decoder_input[:, :] = src[-1, :, :].cuda() # Only for the first timestep
        decoder_input = decoder_input.unsqueeze(0) #.repeat(tgt.shape[0], 1,1)
        decoder_training_input = torch.cat((decoder_input, tgt[:24,]), dim=0)

        for i in range(self.agent_num):
            # Decoder
            decoder_input = torch.zeros(tgt.shape[1], tgt.shape[2]//3).cuda()
            decoder_input[:, :] = src[-1, :, delimiter*i:(delimiter*(i+1))].cuda() # Only for the first timestep
            decoder_training_input = torch.cat((decoder_input.unsqueeze(0), tgt[:24,:,delimiter*i:(delimiter*(i+1))]), dim=0)
            decoder_input = decoder_input.unsqueeze(0)#.repeat(tgt.shape[0], 1,1)
            
            # decoder_training_input = torch.cat((decoder_input, tgt[:24,]), dim=0)
            if inference == True:
                autorgr_out = self.step(decoder_input, tgt, agent_latent[i])
                decoder_output.append(autorgr_out)
            else:
                autorgr_out = self.step(decoder_input, tgt, agent_latent[i])
                decoder_output.append(autorgr_out)
                # decoder_output.append(self.decoder(decoder_training_input, agent_latent[i], tgt_mask=tgt_mask))
                

        tgt = torch.cat(decoder_output, dim=2)
        return tgt
    
    def step(self, decoder_input, tgt, latent,  vis_memory=None, mask=None):
        output_sequence = list()
        decoder_input = decoder_input
        for i in range(tgt.shape[0]):            
            decoder_output = self.decoder(decoder_input, latent, tgt_mask=mask)            
            decoder_output = torch.mean(decoder_output, dim=0).unsqueeze(0)
            output_sequence.append(decoder_output)
            decoder_input = torch.cat(output_sequence, dim=0) #torch.cat(output_sequence, dim=0)                        
            
        tgt = torch.cat(output_sequence, dim=0)
        return tgt

    def compute_attention(self, latent_query, latent_key_1, latent_key_2):
        """
        Computing agent-specific attention
        """
        agent_latent_2, weights_agent_2 = self.agent_attention(latent_query, latent_key_1, latent_key_1)
        agent_latent_3, weights_agent_3 = self.agent_attention(latent_query, latent_key_2, latent_key_2)
        # print (torch.argmax(weights_agent_2[:, -1], dim=1))
        # print (torch.argmax(weights_agent_3[:, -1], dim=1))
        cross_latent_query = torch.cat((latent_query.unsqueeze(0), agent_latent_2.unsqueeze(0), agent_latent_3.unsqueeze(0)), dim=0)
        cross_latent_query = torch.mean(cross_latent_query, dim=0) + latent_query

        return cross_latent_query



    def compute_crossattention(self, latent_query, vis_latent_1, vis_latent_2):
        
        cross_latent_1 = self.agent_attention(latent_query, vis_latent_1, vis_latent_1)[0] 
        cross_latent_query = torch.cat((latent_query.unsqueeze(0), cross_latent_1.unsqueeze(0)), dim=0)
        cross_latent_query = torch.mean(cross_latent_query, dim=0) + latent_query
        
        return cross_latent_query
