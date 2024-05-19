import torch
from torch import nn
from typing import Optional, Tuple, List
from quantize.int_linear import QuantLinear
from quantize.int_matmul import QuantMatMul
import torch.nn.functional as F
from quantize.ptq_norm import PTQLayerNorm
from collections import OrderedDict
import pdb
from models.models_utils import truncate_number
from models.transformation import *


def high_quant(w, wbits,dev):
    maxq = torch.tensor(2 ** wbits - 1)
    shape = w.shape
    w = w.flatten(1)
    tmp = torch.zeros(w.shape[0], device=dev)  # TODO
    wmin = torch.minimum(w.min(1)[0], tmp)
    wmax = torch.maximum(w.max(1)[0], tmp)
    tmp = (wmin == 0) & (wmax == 0)
    wmin[tmp] = -1
    wmax[tmp] = +1
    high_scale = (wmax - wmin) / maxq
    high_zero = torch.round(-wmin / high_scale)
    shape = [-1] + [1] * (len(shape) - 1)
    high_scale = high_scale.reshape(shape)
    high_zero = high_zero.reshape(shape)
    q = torch.clamp(torch.round(w / high_scale) + high_zero, 0, maxq)

    return high_scale * (q - high_zero)

class QuantOPTAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        org_module: nn.Module,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        args=None,
        disable_act_quant=False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )

        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder

        # input is quantized by LayerNorm, set disable_input_quant=True
        self.k_proj = QuantLinear(
            org_module.k_proj,
            args.weight_quant_params,
            args.act_quant_params,
        )
        self.v_proj = QuantLinear(
            org_module.v_proj,
            args.weight_quant_params,
            args.act_quant_params,
        )
        self.q_proj = QuantLinear(
            org_module.q_proj,
            args.weight_quant_params,
            args.act_quant_params,
        )
        self.out_proj = QuantLinear(
            org_module.out_proj, args.weight_quant_params, args.act_quant_params
        )
        self.qkt_matmul = QuantMatMul(
            args.q_quant_params, args.k_quant_params, matmul_func=torch.bmm
        )
        self.pv_matmul = QuantMatMul(
            args.p_quant_params, args.v_quant_params, matmul_func=torch.bmm
        )

        self.use_weight_quant = False
        self.use_act_quant = False

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return (
            tensor.view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        query_states = self.qkt_matmul.quant_x1(query_states)

        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self.k_proj(key_value_states)
            key_states = self.qkt_matmul.quant_x2(key_states)
            key_states = self._shape(key_states, -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            # bsz, seq_len, self.num_heads, self.head_dim -> bsz, self.num_heads, seq_len, self.head_dim
            key_states = self.k_proj(hidden_states)
            key_states = self.qkt_matmul.quant_x2(key_states)
            key_states = self._shape(key_states, -1, bsz)

            value_states = self.v_proj(hidden_states)
            value_states = self.pv_matmul.quant_x2(value_states)
            value_states = self._shape(value_states, -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self.k_proj(hidden_states)
            key_states = self.qkt_matmul.quant_x2(key_states)
            key_states = self._shape(key_states, -1, bsz)

            value_states = self.v_proj(hidden_states)
            value_states = self.pv_matmul.quant_x2(value_states)
            value_states = self._shape(value_states, -1, bsz)
        if self.is_decoder:
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = self.qkt_matmul(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = (
                attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
                + attention_mask
            )
            attn_weights = torch.max(
                attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        # upcast to fp32 if the weights are in fp16. Please see https://github.com/huggingface/transformers/pull/17437
        if attn_weights.dtype == torch.float16:
            attn_weights = nn.functional.softmax(
                attn_weights, dim=-1, dtype=torch.float32
            ).to(torch.float16)
        else:
            attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(
                bsz, self.num_heads, tgt_len, src_len
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(
                bsz, self.num_heads, tgt_len, src_len
            )
            attn_weights = attn_weights_reshaped.view(
                bsz * self.num_heads, tgt_len, src_len
            )
        else:
            attn_probs_reshaped = None

        # attention shape bsz * self.num_heads, tgt_len, src_len
        attn_weights = self.pv_matmul.quant_x1(attn_weights)
        attn_output = self.pv_matmul(attn_weights, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned aross GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_probs_reshaped, past_key_value

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        # setting weight quantization here does not affect actual forward pass
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        for m in self.modules():
            if isinstance(m, (QuantLinear, QuantMatMul)):
                m.set_quant_state(weight_quant, act_quant)


class QuantOPTDecoderLayer(nn.Module):
    def __init__(
        self,
        config,
        ori_layer,
        args,
    ):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = QuantOPTAttention(
            org_module=ori_layer.self_attn,
            embed_dim=self.embed_dim,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            bias=config.enable_bias,
            args=args,
        )
        self.do_layer_norm_before = config.do_layer_norm_before
        self.dropout = config.dropout
        self.self_attn_layer_norm = PTQLayerNorm(
            ori_layer.self_attn_layer_norm
        )
        self.fc1 = QuantLinear(
            ori_layer.fc1,
            weight_quant_params=args.weight_quant_params,
            act_quant_params=args.act_quant_params,
        )
        self.fc2 = QuantLinear(
            ori_layer.fc2,
            weight_quant_params=args.weight_quant_params,
            act_quant_params=args.act_quant_params,
        )
        self.final_layer_norm = PTQLayerNorm(
            ori_layer.final_layer_norm
        )
        self.type = ori_layer.fc1.weight.dtype

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        **kwargs
    ):
        """
        Args:
            hidden_states (`torch.Int8Tensor`): the output of previous layer's layernorm in INT8
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`, *optional*): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        # Self Attention

        residual = hidden_states
        if self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)
        # hidden_states = self.self_attn_layer_norm(hidden_states.float()).to(self.type)

        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )

        hidden_states = nn.functional.dropout(hidden_states, p=0.0, training=False)

        hidden_states = residual + hidden_states

        if not self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        # Fully Connected
        hidden_states_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
        residual = hidden_states

        # residual.add_(hidden_states.to(residual.dtype))
        if self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)
        # hidden_states = self.final_layer_norm(hidden_states.float()).to(self.type)

        
        hidden_states = self.fc1(hidden_states)
        hidden_states = F.relu(hidden_states)

        hidden_states = self.fc2(hidden_states)
        # hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        hidden_states = (residual + hidden_states).view(hidden_states_shape)
        # residual.add_(hidden_states.to(residual.dtype))
        if not self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        # setting weight quantization here does not affect actual forward pass
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        names = []
        for name, m in self.named_modules():
            if isinstance(m, (QuantLinear, QuantMatMul)):
                names.append(name)
                m.set_quant_state(weight_quant, act_quant)

    def binary_temporary(self, wbits, dev, quant_type):
        for name, module in self.named_modules():
            if isinstance(module, QuantLinear):
                module.temp_weight = module.weight
        # quant
        for name, module in self.named_modules():
            if isinstance(module, QuantLinear):
                name_tmp = name.replace(".","_")
                if hasattr(module, "temp_weight"):
                    scale = getattr(self, f"{name_tmp}_scaling_factors")
                    mean = getattr(self, f"{name_tmp}_scaling_means")
                    mask = getattr(self, f"{name_tmp}_mask")
                    r1 = getattr(self, f"{name_tmp}_scaling_rotate_1").to(dev)
                    r2 = getattr(self, f"{name_tmp}_scaling_rotate_2").to(dev)

                    if quant_type == "mix":

                        q_high = high_quant(module.temp_weight.clone(), wbits, dev)
                        q_low = module.temp_weight.clone()
                        q_high = q_high * mask.view(1,-1)

                        q_low = q_low - mean
                        q_sign = q_low.sign()
                        q_low = q_sign * scale
                        q_low += mean
                        q_low = r1@r2 * q_low #r1 * q_low * r2
                        q_low = q_low * ~mask.view(1,-1)

                        module.temp_weight = q_low + q_high
                    else:
                        module.temp_weight = module.temp_weight - mean
                        module.temp_weight = module.temp_weight.sign()
                        module.temp_weight = module.temp_weight * scale
                        module.temp_weight += mean
                        module.temp_weight = r1@r2 * module.temp_weight #r1 * module.temp_weight * r2

                else:
                    scale = getattr(self, f"{name_tmp}_scaling_factors")
                    mean = getattr(self, f"{name_tmp}_scaling_means")
                    mask = getattr(self, f"{name_tmp}_mask")
                    r1 = getattr(self, f"{name_tmp}_scaling_rotate_1")
                    r2 = getattr(self, f"{name_tmp}_scaling_rotate_2")

                    if quant_type == "mix":

                        q_high = high_quant(module.weight.clone(), wbits, dev)
                        q_low = module.weight.clone()
                        q_high = q_high * mask.view(1,-1)

                        q_low = q_low - mean
                        q_sign = q_low.sign()
                        q_low = q_sign * scale
                        q_low += mean
                        q_low = r1@r2 * q_low #r1 * q_low * r2
                        q_low = q_low * ~mask.view(1,-1)

                        module.temp_weight = q_low + q_high
                    else:
                        module.temp_weight = module.weight - mean
                        module.temp_weight = module.temp_weight.sign()
                        module.temp_weight = module.temp_weight * scale
                        module.temp_weight += mean
                        module.temp_weight = r1@r2 * module.temp_weight #r1 * module.temp_weight * r2

                if not hasattr(module, "temp_bias"):
                    module.temp_bias = module.bias
                module.use_temporary_parameter=True

    @torch.no_grad()    
    def binary_inplace(self, wbits, dev, quant_type):
        for name, module in self.named_modules():
            if isinstance(module, QuantLinear):
                # module.weight = module.weight_quantizer(module.weight)
                name_tmp = name.replace(".","_")
                scale = getattr(self, f"{name_tmp}_scaling_factors")
                mean = getattr(self, f"{name_tmp}_scaling_means")
                mask = getattr(self, f"{name_tmp}_mask")
                r1 = getattr(self, f"{name_tmp}_scaling_rotate_1")
                r2 = getattr(self, f"{name_tmp}_scaling_rotate_2")

                if quant_type == "mix":
                    q_high = high_quant(module.weight.clone(), wbits, dev)
                    q_low = module.weight.clone()
                    q_high = q_high * mask.view(1,-1)

                    q_low = q_low - mean
                    q_sign = q_low.sign()
                    q_low = q_sign * scale
                    q_low += mean
                    q_low = r1@r2 * q_low #r1 * q_low * r2
                    q_low = q_low * ~mask.view(1,-1)

                    module.weight = q_low + q_high
                else:
                    module.weight = module.weight - mean
                    module.weight = module.weight.sign()
                    module.weight = module.weight * scale
                    module.weight += mean
                    module.weight = r1@r2 * module.weight #r1 * module.weight *r2

                module.use_temporary_parameter=False
                

    def clear_temp_variable(self):
        for name, module in self.named_modules():
            if isinstance(module, QuantLinear):
                del module.temp_weight
                del module.temp_bias


    def scaling_parameters(self):
        params = []
        template = "scaling_factors" 
        for n, m in self.named_parameters():
            if n.find(template) > -1:
                params.append(m)
        return iter(params) 
    
    def rotation_parameters(self):
        params = []
        template = "scaling_rotate" 
        for n, m in self.named_parameters():
            if n.find(template) > -1:
                params.append(m)
        return iter(params) 
    
    def bias_parameters(self):
        params = []
        template = "scaling_bias" 
        for n, m in self.named_parameters():
            if n.find(template) > -1:
                params.append(m)
        return iter(params)

    def binary_parameters(self):
        params = []
        template = "scaling"
        for n, m in self.named_parameters():
            if n.find(template) > -1:
                params.append(m)
        return iter(params)  
    
    def PTQ_state_dict(self, destination=None, prefix='', keep_vars=False):
        if destination is None:
            destination = OrderedDict()
        for name, param in self.named_parameters():
            if name.find('smooth') > -1 or name.find('bound_factor') > -1:
                destination[prefix + name] = param if keep_vars else param.detach()
        return destination
    

    def register_scales_and_zeros(self):
        for name, module in self.named_modules():
            if isinstance(module, QuantLinear):
                module.weight_quantizer.register_scales_and_zeros()
    
