# Copyright (c) The HuggingFace Inc. team. All rights reserved.
# Copyright (c) Shen Yan. All rights reserved.
# This code is built upon Huggingface's transformers repository.

from transformers import PretrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)


class TCMoEConfig(PretrainedConfig):
    r"""
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    Args:
        vocab_size (`int`, *optional*, defaults to 50_304):
            Vocabulary size of the StableLM model. Defines the number of different tokens that
            can be represented by the `inputs_ids` passed when calling [`StableLMEpochModel`].
        intermediate_size (`int`, *optional*, defaults to 6912):
            Dimension of the MLP representations.
        hidden_size (`int`, *optional*, defaults to 2560):
            Dimension of the decoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1 the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to
            `num_attention_heads`.
        rope_pct (`float`, *optional*, defaults to 1.0):
            Percentage of hidden dimensions to allocate to rotary embeddings.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with.
            Typically set this to something large just in case (e.g., 512 or 1024 or 2048).
        num_experts (`int`, *optional*, defaults to 8):
            Number of experts in the TCMoE layer.
        top_k (`int`, *optional*, defaults to 2):
            Number of top experts to use in the TCMoE layer.
        num_null_experts (`int`, *optional*, defaults to 2):
            Number of null experts in the TCMoE layer.
        initializer_range (`float`, *optional*, defaults to 1e-5):
            The standard deviation of the truncated_normal_initializer for initializing
             all weight matrices.
        norm_eps (`float`, *optional*, defaults to 1e-8):
            The epsilon used by the normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions
            (not used by all models). Only relevant if `config.is_decoder=True`.
        tie_word_embeddings(`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings
    """
    model_type = "tcmoe"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=50432,
        intermediate_size=2816,
        hidden_size=1024,
        num_hidden_layers=32,
        num_attention_heads=16,
        num_key_value_heads=2,
        rope_pct=1.0,
        rope_theta=10000.0,
        max_position_embeddings=2048,
        num_experts=8,
        moe_topk=2,
        num_null_experts=2,
        initializer_range=0.006,
        norm_eps=1e-8,
        use_cache=True,
        bos_token_id=0,
        eos_token_id=0,
        tie_word_embeddings=True,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.intermediate_size = intermediate_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.rope_pct = rope_pct
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.num_experts = num_experts
        self.moe_topk = moe_topk
        self.num_null_experts = num_null_experts
        self.initializer_range = initializer_range
        self.norm_eps = norm_eps
        self.use_cache = use_cache
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )