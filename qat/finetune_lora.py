import sys

sys.path.append(".")
import argparse
import os
import torch
import random
import numpy as np
import torch.nn as nn
import pdb
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    Seq2SeqTrainer
)

# from transformers import LlamaTokenizer, LlamaForCausalLM
from datautils import get_qat_dataset


from transformers import get_cosine_with_hard_restarts_schedule_with_warmup
from peft import (
    get_peft_model,
    LoraConfig,
    PrefixTuningConfig,
    PromptEncoderConfig,
    PromptTuningConfig,
    TaskType,
)
import tensorboard

def get_scheduler(num_training_steps: int):
    def lr_scheduler(optimizer):
        return get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer,
            num_warmup_steps=100,
            num_training_steps=num_training_steps,
            num_cycles=5,
        )

    return lr_scheduler

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def prepare_model_for_training(model):
    for name, param in model.named_parameters():
        # freeze base model's layers
        param.requires_grad = False

    for param in model.parameters():
        if (param.dtype == torch.float16) or (param.dtype == torch.bfloat16):
            param.data = param.data.to(torch.float32)

    # For backward compatibility
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:

        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)

        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()
    return model



def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=False)
    print(f'base model {args.model_id}')
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, device_map="auto", torch_dtype=torch.float16
    )
    if args.ckpt is not None:
        print(f'ckpt {args.ckpt}')
        model.load_state_dict(torch.load(args.ckpt), strict=False)
    # model=model.to_bettertransformer()

    model = prepare_model_for_training(model)
    tokenizer.pad_token = tokenizer.eos_token
    outputs = 'outputs/' + args.model_id.split('/')[-1] + args.ckpt.split('/')[-1].split('.pt')[0]
    print('output to ' + outputs)
    # print("Setup optimizer")
    # opt = torch.optim.AdamW([
    #     p
    #     for p in model.parameters()
    #     if p.requires_grad
    # ], lr=training_args.learning_rate)
    
    # Load dataset
    print("prepare training data")
    data = get_qat_dataset(args.dataset, tokenizer, args.data_percent)
    # pdb.set_trace()

    print("Setup PEFT")
    peft_config = LoraConfig(
        task_type='CAUSAL_LM', inference_mode=False,
        r=64,
        lora_alpha=16, lora_dropout=0.1,
        target_modules=['q_proj','k_proj','v_proj','gate_proj','up_proj','down_proj']
    )
    model = get_peft_model(model, peft_config)
    # Training
    print_trainable_parameters(model)
    # replace_with_qlinear(model)
    # pdb.set_trace()
    # Print mean bit width
    tot_bit=0
    tot_params=0
    
    # for name, module in model.named_modules():
    #     if isinstance(module, BinaryInterface):
    #         module.gen_outlier_mask()
    #         # print(module.outlier_nbits)
    #         tot_bit+=(module.outlier_nbits+1)*module.weight.numel()
    #         tot_params+=module.weight.numel()
    # print(f"mean_bit: {tot_bit/tot_params} frac: {tot_bit/tot_params/16}")

    # Define training arguments
    num_gpus = torch.cuda.device_count()
    per_device_train_batch_size = 1
    gradient_accumulation_steps = 2
    # outputs = 'outputs/llama7b-mix-4-0.1'
    training_args = TrainingArguments(
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=args.train_steps * 0.05,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        bf16=True,
        logging_steps=1,
        max_steps=args.train_steps//(per_device_train_batch_size * gradient_accumulation_steps * num_gpus),
        # num_train_epochs=5,
        output_dir=outputs,
        optim="adamw_torch",
        report_to="tensorboard",
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        train_dataset=data,
        args=training_args,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    
    model.config.use_cache = False

    # Train the model
    trainer.train()

    # Save model
    model.eval()
    save_dir = outputs + f"/{args.train_steps}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # to_regular_linear(model)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"model saved to {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Training Script")
    parser.add_argument(
        "--model_id",
        type=str,
        default="/mnt/data/share/models/llama-7b",
        help="Pretrained model ID",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default='/mnt/data/wangming/root/newBiQuant/result/llama7b.pt',
        help="PTQ ckpt",
    )
    parser.add_argument(
        "--dataset", type=str, default="red_pajama", help="Dataset name"
    )
    parser.add_argument(
        "--data_percent", type=float, default=100, help="Percentage of data to use"
    )
    parser.add_argument(
        "-s", "--train_steps", type=int, default=20000, help="Number of training steps"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Number of training steps"
    )
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    main(args)
