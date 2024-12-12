# PTQ1.61

## Block-wise Optimization and Evaluation
We use LLaMa-7B as an example here:
1. Obtain the channel-wise scales required for initialization:
```
python generate_act_scale_shift.py --model /PATH/TO/LLaMA/llama-7b
```

2. Training and Evaluating PPL
```
CUDA_VISIBLE_DEVICES=0 python main.py --model /PATH/TO/LLAMA/llama-7b --epochs 20 --output_dir ./log/llama-7b --eval_ppl --wbits 4 --abits 16 --quant_type mix --lwc \
--save_dir /CHECKPOINT/TO/FIRST/PTQ \
--calib_dataset wikitext2  \
```

More detailed and optional arguments:
- `--model`: the local model path or huggingface format.
- `--wbits`: weight quantization bits.
- `--quant_type`: quantization type, mix means using structured masks.
- `--lwc`: activate the weight quantizer.
- `--epochs`: training epochs.
- `--nsamples`: number of calibration samples, 128 as default.
- `--eval_ppl`: evaluating the perplexity of quantized models.
- `--multigpu`: to inference larger network on multiple GPUs
- `--save_dir`: saving the quantization model for further exploration.

3. Reproduce the evaluation results of our paper.

1) Download the prebuilt quantized model from our anonymous huggingface repo: https://huggingface.co/ptq161.
2) The detailed reproduction methods please refer to **reproduce.ipynb**.


## Quantization Preprocessing
3. Preprocessing
```
cd preprocessing
CUDA_VISIBLE_DEVICES=0 python restorative_lora.py --model_id /PATH/TO/LLAMA/llama-7b \
--save_dir /CHECKPOINT/TO/FIRST/PTQ

CUDA_VISIBLE_DEVICES=0 python test_perplexity.py  --model_path /PATH/TO/LLAMA/llama-7b \
--ckpt /CHECKPOINT/TO/FIRST/PTQ \
--lora_path ./outputs/CHECKPOINT_NAME/20000-64 \
--output_path /PATH/TO/MERGED/MODEL
```
4. Evaluation PPL after Preprocessing
```
CUDA_VISIBLE_DEVICES=0 python main.py --model /PATH/TO/MERGED/MODEL --epochs 20 --output_dir ./log/llama-7b --eval_ppl --wbits 4 --abits 16 --quant_type mix --lwc \
--save_dir /CHECKPOINT/TO/SECOND/PTQ \
--calib_dataset wikitext2  \
```

## Reasoning Tasks Evaluation

Please follow lm-eval-harness for evaluating Hellaswag, PIQA, MMLU, GSM8K, LAMBADA, etc. 

'lm_eval' file is lm-evaluation-harness, a open-sourced evaluation framework from https://github.com/EleutherAI/lm-evaluation-harness, contains datasets, benchmarks, etc.