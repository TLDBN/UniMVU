"""Canonical public UniMVU model."""

from transformers import AutoConfig, AutoModelForCausalLM

from ._unimvu_base import UniMVUQwen2Model
from ._unimvu_mix import UniMVUUnifiedConfig, UniMVUUnifiedForCausalLM


class UniMVUConfig(UniMVUUnifiedConfig):
    """Public configuration for the released UniMVU model."""

    model_type = "unimvu"


class UniMVUModel(UniMVUQwen2Model):
    config_class = UniMVUConfig


class UniMVUForCausalLM(UniMVUUnifiedForCausalLM):
    """Released UniMVU causal LM using the mixed-modality-safe implementation."""

    config_class = UniMVUConfig


try:
    AutoConfig.register(UniMVUConfig.model_type, UniMVUConfig)
except ValueError:
    pass

AutoModelForCausalLM.register(UniMVUConfig, UniMVUForCausalLM)
