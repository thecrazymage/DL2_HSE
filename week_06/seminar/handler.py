import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from collections import defaultdict
from diffusers import StableDiffusionXLPipeline
from diffusers.models.attention_processor import AttnProcessor2_0, Attention

class CustomAttentionProcessor(AttnProcessor2_0):
    def __init__(self, handler, layer_name):
        super().__init__()
        self.layer_name = layer_name
        self.handler = handler
        
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs
    ):
        
        residual = hidden_states
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        
        self.handler.queries[self.layer_name].append(query.detach().cpu())
        self.handler.keys[self.layer_name].append(key.detach().cpu())
        
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states
    
    
class Handler:
    def __init__(self, pipeline: StableDiffusionXLPipeline):
        self.pipeline = pipeline
        
        self.queries = defaultdict(list)
        self.keys = defaultdict(list)
        # self.values = defaultdict(list)
        
    def init_attention_processors(self, reset=False):
        attn_procs = {}
        custom = []
    
        for name in self.pipeline.unet.attn_processors.keys():
            is_cross_attention = 'attn2' in name
            
            if is_cross_attention and not reset:
                attn_procs[name] = CustomAttentionProcessor(self, name)
                custom.append(name)
            else:
                attn_procs[name] = AttnProcessor2_0()
                
        self.pipeline.unet.set_attn_processor(attn_procs)
        return custom

    def register(self):
        return self.init_attention_processors()

    def remove(self):
        self.init_attention_processors(reset=True)
