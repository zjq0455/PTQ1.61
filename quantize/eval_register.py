import torch
import torch.nn as nn
from models.int_llama_layer import QuantLlamaDecoderLayer
from models.int_opt_layer import QuantOPTDecoderLayer
from models.int_falcon_layer import QuantFalconDecoderLayer
from quantize.int_linear import QuantLinear
import auto_gptq.nn_modules.qlinear.qlinear_cuda as qlinear_cuda
from contextlib import nullcontext
import copy
import math
import utils
import os
import pdb
import gc
import numpy as np



def get_named_linears(module):
    return {name: m for name, m in module.named_modules() if isinstance(m, QuantLinear)}


def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


def blockptq(
    lm,
    args,
    dataloader,
    act_scales,
    act_shifts,
    logger=None,
):
    logger.info("Starting ...")
    
    # move embedding layer and first layer to target device
    model = lm.model
    dev = lm.device
    use_cache = model.config.use_cache
    model.config.use_cache = False
    is_llama = False
    if "llama" in args.net.lower():
        is_llama = True
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
        DecoderLayer = QuantLlamaDecoderLayer
        pairs = {
            "q_proj":"qkv",
            "o_proj":"out",
            "up_proj":"fc1"
        }
        layer_name_prefix = "model.layers"
    elif "opt" in args.net.lower():
        layers = model.model.decoder.layers
        print(layers)
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
        if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
            model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
        if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
        DecoderLayer = QuantOPTDecoderLayer
        pairs = {
            "q_proj":"qkv",
            "out_proj":"out",
            "fc1":"fc1"
        }
        layer_name_prefix = "model.decoder.layers"
    elif "falcon" in args.net.lower():
        layers = model.transformer.h
        model.transformer.word_embeddings.to(dev)
        model.transformer.ln_f.to(dev)
        model.lm_head.to(dev)
        DecoderLayer = QuantFalconDecoderLayer
        layer_name_prefix = "model.transformer.h"
    else:
        raise ValueError("Only support for opt/llama/Llama-2/falcon now")
    
    
    layers[0] = layers[0].to(dev)
    print("------------")
    if args.deactive_amp and args.epochs>0:
        dtype = torch.float
        traincast = nullcontext
    else:
        dtype = torch.float16
        traincast = torch.cuda.amp.autocast
    inps = torch.zeros(
        (args.nsamples, lm.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )

    cache = {"i": 0}
    # catch the first layer input 第一层的输入
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.is_llama = False

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            if self.is_llama:
                cache["position_ids"] = kwargs["position_ids"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    layers[0].is_llama = is_llama

    with torch.no_grad():
        for batch in dataloader:
            if cache["i"] >= args.nsamples:
                break
            try:
                model(batch[0].to(dev))
            except ValueError:
                pass
    
    # move embedding layer and first layer to cpu
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    if "llama" in args.net.lower():
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()
    elif "opt" in args.net.lower():
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
        if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
            model.model.decoder.project_out = model.model.decoder.project_out.cpu()
        if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    elif 'falcon' in args.model:
        model.transformer.word_embeddings =  model.transformer.word_embeddings.cpu()
    else:
        raise ValueError("Only support for opt/llama/Llama-2/falcon now")
    torch.cuda.empty_cache()
    

    attention_mask = cache["attention_mask"]
    # attention_mask_batch = attention_mask.repeat(args.batch_size,1,1,1) if args.deactive_amp else attention_mask.repeat(args.batch_size,1,1,1).float()
    loss_func = torch.nn.MSELoss()
    if is_llama:
        position_ids = cache["position_ids"]
    else:
        position_ids = None
    cossim = nn.CosineSimilarity(dim=2)
    
    for i in range(len(layers)):
        logger.info(f"=== Start quantize layer {i} ===")
        print(len(layers))
        
        layer = layers[i].to(dev)
        qlayer = DecoderLayer(lm.model.config, layer, args)
        qlayer = qlayer.to(dev)
        
        # obtain output of full-precision model
        qlayer.set_quant_state(weight_quant=False, act_quant=False)

        qlayer.set_quant_state(weight_quant=False, act_quant=True)  # weight will be manually quantized before forward
        
        for name,module in qlayer.named_modules():
            if isinstance(module, QuantLinear):
                weight = module.weight.clone()
                act = act_scales[f"{layer_name_prefix}.{i}.{name}"].to(device=dev, dtype=dtype).clamp(min=1e-5) #4096
                mask = torch.zeros(1, weight.shape[1], dtype=torch.bool)
                # saliency = torch.mean(torch.abs(weight),dim=1)
                thresh = torch.sort(act.flatten(),descending=True)[0][int(act.numel() * 0.2)] #0.1
                mask = act >= thresh

                # init scaling factors and means.

                if args.quant_type == "low":

                    w_mean = weight.mean(1).view(-1, 1) #按行求  # 1*4096
                    w_mean = torch.where(torch.isnan(w_mean), torch.zeros_like(w_mean), w_mean)
                    # w = torch.where(w!=0, w - w_mean.unsqueeze(-1), w)
                    scale = weight.abs().mean(1,keepdim=True) #也是按行求
                    scale = torch.where(torch.isnan(scale), torch.zeros_like(scale), scale) #1*4096
                else:

                    w = weight * ~(mask.view(1,-1))
                    w_nonzero_mean = (w * (w != 0).float()).sum(dim=1) / (w != 0).sum(dim=1).float() #按行求
                    w_nonzero_mean = torch.where(torch.isnan(w_nonzero_mean), torch.zeros_like(w_nonzero_mean), w_nonzero_mean)
                    scale = ((w * (w != 0)).abs().sum(dim=1) / (w != 0).sum(dim=1).float()).view(-1, 1) #也是按行求 (w * (w != 0)
                    scale = torch.where(torch.isnan(scale), torch.zeros_like(scale), scale) 
                    w_mean = w_nonzero_mean.view(-1, 1)

                name_tmp = name.replace(".","_")

                r1 = torch.ones(weight.shape[0], 1).to(dev)
                r2 = torch.ones(1, weight.shape[1]).to(dev)

                qlayer.register_parameter(f"{name_tmp}_scaling_factors",torch.nn.Parameter(scale))
                qlayer.register_parameter(f"{name_tmp}_scaling_rotate_1",torch.nn.Parameter(r1))
                qlayer.register_parameter(f"{name_tmp}_scaling_rotate_2",torch.nn.Parameter(r2))
                qlayer.register_parameter(f"{name_tmp}_scaling_means",torch.nn.Parameter(w_mean,requires_grad=False)) # ,requires_grad=False
                qlayer.register_parameter(f"{name_tmp}_mask",torch.nn.Parameter(mask,requires_grad=False))
    
        
        # real smooth and quantization
        qlayer.binary_inplace(args.wbits, dev, args.quant_type)       
        if args.epochs>0:
            qlayer.half()
            layers[i] = qlayer.to("cpu")
        else:
            # qlayer.register_scales_and_zeros()
            qlayer.half()
            layers[i] = qlayer.to("cpu")       
        
        del layer
        torch.cuda.empty_cache()

    del inps
    torch.cuda.empty_cache()
    gc.collect()                    
    model.config.use_cache = use_cache
    return model