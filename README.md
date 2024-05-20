# PTQ1.61

## Block-wise Optimization and Evaluation
We use LLaMa-7B as an example here:
1. Obtain the channel-wise scales required for initialization:
```
python generate_act_scale_shift.py --model /PATH/TO/LLaMA/llama-7b
```

2. Training and Evaluation
```
CUDA_VISIBLE_DEVICES=0 python main.py --model /PATH/TO/LLAMA/llama-7b --epochs 20 --output_dir ./log/llama-7b --eval_ppl --wbits 4 --abits 16 --quant_type mix --lwc \
--save_dir /CHECKPOINT/TO/FIRST/PTQ \
--calib_dataset wikitext2  \
--tasks piqa,arc_easy,arc_challenge,boolq,hellaswag,winogrande
```

## Quantization Preprocessing
3. Restorative LoRA
```
cd qat
CUDA_VISIBLE_DEVICES=0 python finetune_lora.py --model_id /PATH/TO/LLAMA/llama-7b \
--save_dir /CHECKPOINT/TO/FIRST/PTQ --lora_r 64 -s 20000
```
5. Merge with Quantized Model
```
CUDA_VISIBLE_DEVICES=0 python test_perplexity.py  --model_path /PATH/TO/LLAMA/llama-7b \
--ckpt /CHECKPOINT/TO/FIRST/PTQ \
--lora_path ./outputs/CHECKPOINT_NAME/20000-64 \
--output_path /PATH/TO/MERGED/MODEL
```
6. PTQ and Evaluation
```
CUDA_VISIBLE_DEVICES=0 python main.py --model /PATH/TO/MERGED/MODEL --epochs 20 --output_dir ./log/llama-7b --eval_ppl --wbits 4 --abits 16 --quant_type mix --lwc \
--save_dir /CHECKPOINT/TO/SECOND/PTQ \
--calib_dataset wikitext2  \
--tasks piqa,arc_easy,arc_challenge,boolq,hellaswag,winogrande
```

More detailed and optional arguments:
- `--model`: the local model path or huggingface format.
- `--wbits`: weight quantization bits.
- `--quant_type`: quantization type.
- `--lwc`: activate the weight quantizer.
- `--epochs`: training epochs.
- `--nsamples`: number of calibration samples, 128 as default.
- `--eval_ppl`: evaluating the perplexity of quantized models.
- `--tasks`: evaluating zero-shot tasks.
- `--multigpu`: to inference larger network on multiple GPUs
- `--save_dir`: saving the quantization model for further exploration.

## Related Project

[GPTQ: Accurate Post-training Compression for Generative Pretrained Transformers](https://github.com/IST-DASLab/gptq)

[OmniQuant: Omnidirectionally Calibrated Quantization for Large Language Models](https://arxiv.org/abs/2308.13137)

[PB-LLM: Partially Binarized Large Language Models](https://github.com/hahnyuan/PB-LLM)

[BiLLM: Pushing the Limit of Post-Training Quantization for LLMs](https://github.com/Aaronhuang-778/BiLLM)