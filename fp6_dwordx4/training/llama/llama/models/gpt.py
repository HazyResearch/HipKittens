import logging
import math
import re
from collections import namedtuple
from functools import partial

import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config

from .block import Block
from .embedding import GPT2Embeddings
from .mha import MHA
from .mlp import GatedMlp
from llama.ops.triton.layer_norm import RMSNorm

logger = logging.getLogger(__name__)
from .utils.hf import load_config_hf, load_state_dict_hf

class GPT2MixerConfig(GPT2Config):
    def __init__(self, *args, **kwargs):
        self.mixer = kwargs.pop("mixer", None)
        super().__init__(*args, **kwargs)

def create_mha_cls(config, layer_idx=None, device=None, dtype=None):
    factory_kwargs = {"device": device, "dtype": dtype}

    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    softmax_scale = 1.0 if not config.scale_attn_weights else head_dim ** (-0.5)
    if config.scale_attn_by_inverse_layer_idx:
        assert layer_idx is not None
        softmax_scale /= float(layer_idx + 1)
    qkv_proj_bias = getattr(config, "qkv_proj_bias", True)
    out_proj_bias = getattr(config, "out_proj_bias", True)
    rotary_emb_dim = int(getattr(config, "rotary_emb_fraction", 0.0) * head_dim)
    rotary_emb_base = getattr(config, "rotary_emb_base", 10000.0)
    rotary_emb_scale_base = getattr(config, "rotary_emb_scale_base", None)
    rotary_emb_interleaved = getattr(config, "rotary_emb_interleaved", False)
    use_aiter_attn = getattr(config, "use_aiter_attn", False)
    use_hip_attn = getattr(config, "use_hip_attn", False)
    fused_bias_fc = getattr(config, "fused_bias_fc", False)
    mha_cls = MHA
    serial_kwargs = (
        {"fused_bias_fc": fused_bias_fc} 
    )
    num_heads_kv = getattr(config, "n_head_kv", None)
    mixer_cls = partial(
        mha_cls,
        num_heads=config.num_attention_heads,
        num_heads_kv=num_heads_kv,
        qkv_proj_bias=qkv_proj_bias,
        out_proj_bias=out_proj_bias,
        dropout=config.attn_pdrop,
        softmax_scale=softmax_scale,
        causal=True,
        layer_idx=layer_idx,
        rotary_emb_dim=rotary_emb_dim,
        rotary_emb_base=rotary_emb_base,
        rotary_emb_scale_base=rotary_emb_scale_base,
        rotary_emb_interleaved=rotary_emb_interleaved,
        use_aiter_attn=use_aiter_attn,
        use_hip_attn=use_hip_attn,
        **serial_kwargs,
        **factory_kwargs,
    )
    return mixer_cls


def create_mlp_cls(config, layer_idx=None, device=None, dtype=None):
    factory_kwargs = {"device": device, "dtype": dtype}
    mlp_fc1_bias = getattr(config, "mlp_fc1_bias", True)
    mlp_fc2_bias = getattr(config, "mlp_fc2_bias", True)
    mlp_cls = partial(
        GatedMlp,
        hidden_features=config.n_inner,
        activation=F.silu,
        bias1=mlp_fc1_bias,
        bias2=mlp_fc2_bias,
        **factory_kwargs,
    )   
    return mlp_cls


def create_block(config, layer_idx=None, device=None, dtype=None, **kwargs):
    factory_kwargs = {"device": device, "dtype": dtype}
    mixer_cls = create_mha_cls(config, layer_idx, **factory_kwargs)
    mlp_cls = create_mlp_cls(config, layer_idx, **factory_kwargs)
    use_rms_norm = getattr(config, "rms_norm", False)
    norm_cls = partial(
        nn.LayerNorm if not use_rms_norm else RMSNorm,
        eps=config.layer_norm_epsilon,
        **factory_kwargs,
    )
    # TD [2022-07-30]: Force residual in fp32, seems to make fp16 training more stable
    residual_in_fp32 = getattr(config, "residual_in_fp32", False)
    resid_dropout1 = config.resid_pdrop if layer_idx is None or layer_idx > 0 else config.embd_pdrop
    prenorm = getattr(config, "prenorm", True)
    
    block = Block(
        config.hidden_size,
        mixer_cls,
        mlp_cls,
        norm_cls=norm_cls,
        prenorm=prenorm,
        resid_dropout1=resid_dropout1,
        resid_dropout2=config.resid_pdrop,
        fused_dropout_add_ln=False,
        residual_in_fp32=residual_in_fp32,
        layer_idx=layer_idx,
    )

    block.layer_idx = layer_idx
    return block

class GPTPreTrainedModel(nn.Module):
    """An abstract class to handle weights initialization and
    a simple interface for dowloading and loading pretrained models.
    """

    def __init__(self, config, *inputs, **kwargs):
        super().__init__()
        if not isinstance(config, GPT2Config):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `GPT2Config`. "
                "To create a model from a Google pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                )
            )
        self.config = config

    @classmethod
    def from_pretrained_hf(cls, pretrained_model_name, device=None, **kwargs):
        config_data = load_config_hf(pretrained_model_name)
        config = GPT2Config(**config_data)
        model = cls(config, device=device, **kwargs)
        state_dict = load_state_dict_hf(pretrained_model_name, device=device)
        # remove the 'model.' prefix from the keys
        state_dict = {re.sub("^model\.", "", k): v for k, v in state_dict.items()}
        # remove Unexpected key(s) in state_dict
        state_dict = {k: v for k, v in state_dict.items() if "metrics" not in k}

        model.load_state_dict(state_dict)
        return model.to(device=device)


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(module, n_layer, initializer_range=0.02, rescale_prenorm_residual=True, use_weight_init=True):
    if isinstance(module, nn.Linear):
        if use_weight_init:
            nn.init.normal_(module.weight, std=initializer_range)   # SA: this line isn't in Mamba init code
        else:
            print(f"Skipping!")
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                nn.init.normal_(p, mean=0.0, std=initializer_range / math.sqrt(2 * n_layer))


class GPTModel(GPTPreTrainedModel):
    def __init__(self, config: GPT2Config, device=None, dtype=None):
        super().__init__(config)
        factory_kwargs = {"device": device, "dtype": dtype}
        assert config.activation_function in ["swiglu"]
        pad_vocab_size_multiple = getattr(config, "pad_vocab_size_multiple", 1)
        vocab_size = (
            math.ceil(config.vocab_size / pad_vocab_size_multiple) * pad_vocab_size_multiple
        )
        # TD [2022-07-30]: Force residual in fp32, seems to make fp16 training more stable
        self.residual_in_fp32 = getattr(config, "residual_in_fp32", False)
        # These 2 options are for OPT-350m
        self.prenorm = getattr(config, "prenorm", True)
        use_rms_norm = getattr(config, "rms_norm", False)
        word_embed_proj_dim = getattr(config, "word_embed_proj_dim", None)
        # For GPT-J, GPT-NeoX

        self.embeddings = GPT2Embeddings(
            config.hidden_size,
            vocab_size,
            config.max_position_embeddings,
            word_embed_proj_dim=word_embed_proj_dim,
            **factory_kwargs,
        )

        # We change the order of dropout, residual and layer norm:
        # Instead of LN -> Attn / MLP -> Dropout -> Add, we do:
        # Dropout -> Add -> LN -> Attn / MLP, returning both the residual branch (output of Add) and
        # the main branch (output of MLP). The model definition is unchanged, but the mapping of the
        # nn.Dropout probabilities are changed.
        # This is for performance reason: we can fuse dropout + add + layer_norm.
        self.layers = nn.ModuleList(
            [
                create_block(config, layer_idx=i, **factory_kwargs)
                for i in range(config.num_hidden_layers)
            ]
        )
        self.fused_dropout_add_ln = False
        if self.prenorm:
            self.drop_f = nn.Dropout(config.resid_pdrop)
            norm_cls = nn.LayerNorm if not use_rms_norm else RMSNorm
            self.ln_f = norm_cls(
                config.hidden_size, eps=config.layer_norm_epsilon, **factory_kwargs
            )
        
        if getattr(config, "special_initializer", False):
            initializer_range = (2 / (config.n_embd * 5)) ** 0.5
        else:
            initializer_range = config.initializer_range

        self.apply(
            partial(
                _init_weights,
                n_layer=config.num_hidden_layers,
                initializer_range=config.initializer_range,
                use_weight_init=getattr(config, "use_weight_init", True),
            )
        )
        self.tie_weights()

    def tie_weights(self):
        pass

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward(self, input_ids, position_ids=None, inference_params=None, attention_mask=None, stream=None):
        embedding_kwargs = ({})
        
        assert input_ids is not None, "Input tensor input_ids is None"
        hidden_states = self.embeddings(input_ids, position_ids=position_ids, **embedding_kwargs)
        assert hidden_states is not None, "Hidden states are None"
        
        residual = None
        mixer_kwargs = ({})
        if inference_params is not None:
            mixer_kwargs["inference_params"] = inference_params
            mixer_kwargs['stream'] = stream
        
        for layer in self.layers:
            if self.prenorm:
                assert hidden_states is not None, "Hidden states are None"
                hidden_states, residual = layer(hidden_states, residual=residual, mixer_kwargs=mixer_kwargs)
                assert hidden_states is not None, "Hidden states are None"
            else:
                assert hidden_states is not None, "Hidden states are None"
                hidden_states = layer(hidden_states, position_ids=position_ids, mixer_kwargs=mixer_kwargs)
                assert hidden_states is not None, "Hidden states are None"
        if self.prenorm:
            if not self.fused_dropout_add_ln:
                assert hidden_states is not None, "Hidden states are None"
                dropped = self.drop_f(hidden_states)
                assert dropped is not None, "Dropped states are None"
                residual = (dropped + residual) if residual is not None else dropped
                assert residual is not None, "Residual states are None"
                hidden_states = self.ln_f(residual.to(dtype=self.ln_f.weight.dtype))
                assert hidden_states is not None, "Hidden states are None"
            else:
                assert False, "Fused dropout add layer norm is not supported"
        return hidden_states


class GPTLMHeadModel(GPTPreTrainedModel):
    def __init__(self, config: GPT2Config, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(config)
        self.transformer = GPTModel(config,  **factory_kwargs)
        self.tie_word_embeddings = getattr(config, "tie_word_embeddings", True)
        lm_head_bias = getattr(config, "lm_head_bias", False)
        pad_vocab_size_multiple = getattr(config, "pad_vocab_size_multiple", 1)
        vocab_size = (
            math.ceil(config.vocab_size / pad_vocab_size_multiple) * pad_vocab_size_multiple
        )
        # This option is for OPT-350m
        word_embed_proj_dim = getattr(config, "word_embed_proj_dim", None)
        embed_dim = config.n_embd if word_embed_proj_dim is None else word_embed_proj_dim
        if word_embed_proj_dim is not None:
            self.project_out = nn.Linear(config.n_embd, embed_dim, bias=False, **factory_kwargs)
        else:
            self.project_out = None
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=lm_head_bias, **factory_kwargs)

        # Initialize weights and apply final processing
        self.apply(
            partial(
                _init_weights,
                n_layer=config.num_hidden_layers,
                initializer_range=config.initializer_range,
            )
        )
        self.tie_weights()
        
    def tie_weights(self):
        if self.tie_word_embeddings:
            self.lm_head.weight = self.transformer.embeddings.word_embeddings.weight

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.transformer.allocate_inference_cache(
            batch_size, max_seqlen, dtype=dtype, **kwargs
        )

    def forward(self, input_ids, position_ids=None, inference_params=None, num_last_tokens=0, stream=None, **kwargs):
        """
        input_ids: (batch, seqlen) int tensor
        inference_params: for generation. Adapted from Megatron-LM (and Apex)
        https://github.com/NVIDIA/apex/blob/3ff1a10f72ec07067c4e44759442329804ac5162/apex/transformer/testing/standalone_transformer_lm.py#L470
        num_last_tokens: if > 0, only return the logits for the last n tokens
        """
        if type(input_ids) == list:
            input_ids = input_ids[0]    
        assert (
            input_ids.ndim == 2
        ), f"Expected `input_ids` to have shape [b, slen], but got shape {input_ids.shape}"
        b, slen = input_ids.shape
        
        assert input_ids is not None, "Input tensor input_ids is None"
        hidden_states = self.transformer(
            input_ids, position_ids=position_ids, inference_params=inference_params,
            stream=stream
        )
        assert hidden_states is not None, "Hidden states are None"
        
        if inference_params is not None:
            assert hidden_states.ndim == 3, "sequence_parallel is not supported in generation mode"
        if num_last_tokens > 0:
            hidden_states = hidden_states[:, -num_last_tokens:]
        if self.project_out is not None:
            hidden_states = self.project_out(hidden_states)
        lm_logits = self.lm_head(hidden_states)
        # During inference, we want the full logit for sampling
        CausalLMOutput = namedtuple("CausalLMOutput", ["logits"])
        return CausalLMOutput(logits=lm_logits)

    def load_state_dict(self, state_dict, strict=True):
        # Remapping from our checkpoints that used a different ordering of layers in the block
        # Previous: Attn / MLP -> Dropout -> Add -> LN
        # Current: Dropout -> Add -> LN -> Attn / MLP
        if "transformer.ln_0.weight" in state_dict:
            n_layers = len(self.transformer.layers)
            ln_weight = state_dict.pop(f"transformer.layers.{n_layers - 1}.norm2.weight")
            ln_bias = state_dict.pop(f"transformer.layers.{n_layers - 1}.norm2.bias")
            state_dict["transformer.ln_f.weight"] = ln_weight
            state_dict["transformer.ln_f.bias"] = ln_bias
            for l in reversed(range(n_layers)):
                ln_weight = state_dict.pop(f"transformer.layers.{l}.norm1.weight")
                ln_bias = state_dict.pop(f"transformer.layers.{l}.norm1.bias")
                state_dict[f"transformer.layers.{l}.norm2.weight"] = ln_weight
                state_dict[f"transformer.layers.{l}.norm2.bias"] = ln_bias
                if l > 0:
                    ln_weight = state_dict.pop(f"transformer.layers.{l - 1}.norm2.weight")
                    ln_bias = state_dict.pop(f"transformer.layers.{l - 1}.norm2.bias")
                    state_dict[f"transformer.layers.{l}.norm1.weight"] = ln_weight
                    state_dict[f"transformer.layers.{l}.norm1.bias"] = ln_bias
            ln_weight = state_dict.pop("transformer.ln_0.weight")
            ln_bias = state_dict.pop("transformer.ln_0.bias")
            state_dict[f"transformer.layers.0.norm1.weight"] = ln_weight
            state_dict[f"transformer.layers.0.norm1.bias"] = ln_bias
        return super().load_state_dict(state_dict, strict=strict)