from collections import defaultdict
import copy
import json
import os
from pp_utils import get_loaders
from os.path import exists, join, isdir
from dataclasses import dataclass, field
import sys
from typing import Optional, Dict, Sequence
import numpy as np
from tqdm import tqdm
import logging
from torch import nn
import torch
import pdb
import transformers
from torch.nn.utils.rnn import pad_sequence
import argparse
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    set_seed, 
    Seq2SeqTrainer,
    LlamaTokenizerFast
)
from datasets import load_dataset

from peft import (
    LoraConfig,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    PeftModel
)
import tensorboard
torch.backends.cuda.matmul.allow_tf32 = True

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"

def get_accelerate_model(model_path, ckpt_path=None, lora_path=None, output_path=None):
    print(f'loading base model {model_path}...')
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map="auto", torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if ckpt_path:
        print(f'loading ckpt {ckpt_path}...')
        model.load_state_dict(torch.load(ckpt_path), strict=False)
    if lora_path is not None:
        print(f'loading lora adpater {lora_path}...')
        model = PeftModel.from_pretrained(model, lora_path)
        model = model.merge_and_unload()
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
    return model
    
def evaluate(model, tokenizer_path, logger):
    results = {}
    seqlen = 2048
    seed = 42
    if True:
        # for dataset in ["wikitext2", "ptb", "c4","ptb-new",'c4-new']:
        for dataset in ["wikitext2", "c4"]:
            cache_testloader = f'/root/newBiQuant/cache/testloader_opt_{dataset}_all.cache'   #opt
            if os.path.exists(cache_testloader):
                testloader = torch.load(cache_testloader)
                logger.info(f"load calibration from {cache_testloader}")
            else:
                dataloader, testloader = get_loaders(
                    dataset,
                    seed=seed,
                    model=tokenizer_path,
                    seqlen=seqlen,
                )
                torch.save(testloader, cache_testloader)
            if "c4" in dataset:
                testenc = testloader
            else:
                testenc = testloader.input_ids

            nsamples = testenc.numel() // seqlen
            use_cache = model.config.use_cache
            model.config.use_cache = False
            model.eval()
            nlls = []
            for i in tqdm(range(nsamples)):
                batch = testenc[:, (i * seqlen) : ((i + 1) * seqlen)].cuda()
                # pdb.set_trace()
                # logits = model(batch)['logits']
                outputs = model.model(batch)
                logits = outputs[0]
                logits = model.lm_head(logits)
                shift_logits = logits[:, :-1, :]
                shift_labels = testenc[:, (i * seqlen) : ((i + 1) * seqlen)][
                    :, 1:
                ].to(model.lm_head.weight.device)
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )
                neg_log_likelihood = loss.float() * seqlen
                nlls.append(neg_log_likelihood)

            ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))
            logger.info(f'{dataset} : {ppl.item()}')
            model.config.use_cache = use_cache
            results[dataset] = ppl.item()
    return results

if __name__ == "__main__":
    model_path = "/root/myModels/opt-2.7b"
    lora_path = "/root/newBiQuant/qat/20000"
    output_path = "/root/newBiQuant/qat/opt-2.7b_merged_mix_4_current-20000"
    ckpt = "/root/newBiQuant/log/opt-2.7b/mix_4_current.pth"
    model = get_accelerate_model(model_path, ckpt, lora_path, output_path)
    model.eval()
    # import pdb
    # pdb.set_trace()s
    print(model.device)
    for n,p in model.named_parameters():
        p.requires_grad = False
    results = evaluate(model ,model_path, logger)
    print('perplexity result:')
    for k,v in results.items():
        print(k, v)