import copy
import logging
import math
import re
import os
import warnings
from datetime import timedelta
from typing import List, Optional, Tuple, Union

import numpy as np
import PIL
import torch
import transformers
from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs
from accelerate.state import AcceleratorState
from decord import VideoReader
import decord
from packaging import version
from tqdm import tqdm
from transformers import BitsAndBytesConfig, AutoTokenizer
from libs.dataset.video_loading_utils import fps_base_temporal_sampling
from einops import rearrange, repeat
from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.load_video import read_video_pyav
from libs.model import *
from libs.model.multimodal_LMM._unimvu_base import (
    UniMVUQwen2Config,
    UniMVUQwen2ForCausalLM,
)
from libs.model.multimodal_LMM._unimvu_mix import (
    UniMVUUnifiedConfig,
    UniMVUUnifiedForCausalLM,
)
from libs.utils.video_io import SafeVideoReader

# Suppress warnings
warnings.filterwarnings("ignore")

# Configure logging
eval_logger = logging.getLogger("lmms-eval")

# Enable TF32 for CUDA
torch.backends.cuda.matmul.allow_tf32 = True

# Import LLaVA modules
try:
    from libs.constants import (
        DEFAULT_IM_END_TOKEN,
        DEFAULT_IM_START_TOKEN,
        DEFAULT_IMAGE_TOKEN,
        IGNORE_INDEX,
        IMAGE_TOKEN_INDEX,
    )
    from libs.conversation_lib import SeparatorStyle, conv_templates
    from libs.mm_utils import (
        KeywordsStoppingCriteria,
        get_model_name_from_path,
        process_images,
        tokenizer_image_token,
    )
    #from llava.model.builder import load_pretrained_model
except ImportError as e:
    eval_logger.debug(f"LLaVA is not installed. Please install LLaVA to use this model.\nError: {e}")


# Determine best attention implementation
if version.parse(torch.__version__) >= version.parse("2.1.2"):
    best_fit_attn_implementation = "sdpa"
else:
    best_fit_attn_implementation = "eager"


from libs.utils.train_utils import MODEL_ARGUMENTS_MAPPING, DATA_ARGUMENTS_MAPPING

BASE_MODEL_TYPE = "unimvu"
UNIFIED_MODEL_TYPE = "unimvu_uni"


def _resolve_unimvu_eval_classes(requested_model_type: str):
    requested = (requested_model_type or BASE_MODEL_TYPE).lower()

    if requested == UNIFIED_MODEL_TYPE:
        return UniMVUUnifiedForCausalLM, UniMVUUnifiedConfig
    if requested == BASE_MODEL_TYPE:
        return UniMVUQwen2ForCausalLM, UniMVUQwen2Config

    raise ValueError(
        f"Unsupported UniMVU model type '{requested_model_type}'. "
        f"Use '{BASE_MODEL_TYPE}' or '{UNIFIED_MODEL_TYPE}'."
    )


def _natural_key(value: str):
    """Helper for natural sorting of frame filenames."""
    return [int(chunk) if chunk.isdigit() else chunk.lower() for chunk in re.split(r"(\d+)", value)]
# temporal_aggregator_parameters = [name for name, _ in opt_model.named_parameters() if "temporal_aggregator" in name]


def load_trained_model_for_eval_unimvuv(
    model_path,
    model_base,
    model_name,
    model_arg_name='VideoFeatModelArgumentsUniMVU',
    data_arg_name='default',
    load_8bit=False,
    load_4bit=False,
    device_map="auto",
    device="cuda",
    use_flash_attn=False,
    **kwargs,
):
    if model_path is None:
        raise ValueError("model_path must be provided for UniMVU evaluation.")

    kwargs = {"device_map": device_map, **kwargs}

    if model_base is None:
        raise ValueError("model_base must be provided for UniMVU evaluation.")

    if device != "cuda":
        kwargs['device_map'] = {"": device}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    if use_flash_attn:
        kwargs['attn_implementation'] = 'flash_attention_2'

    tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
    default_model_arg = MODEL_ARGUMENTS_MAPPING[model_arg_name]
    default_data_args = DATA_ARGUMENTS_MAPPING[data_arg_name]
    requested_model_type = str(getattr(default_model_arg, "model_type", BASE_MODEL_TYPE)).lower()
    model, model_config = _resolve_unimvu_eval_classes(requested_model_type)

    print(
        'Warning: we are using MODEL_ARGUMENTS_MAPPING:',
        model.__name__,
        'DATA_ARGUMENTS_MAPPING:',
        model_config.__name__,
        'requested_model_type:',
        requested_model_type,
    )

    lora_cfg_pretrained = model_config.from_pretrained(model_path)
    lora_cfg_pretrained.mm_video_tower = None

    for key in default_model_arg.__dict__:
        if not key.startswith('__'):
            if not hasattr(lora_cfg_pretrained, key):
                setattr(lora_cfg_pretrained, key, default_model_arg.__dict__[key])

    if getattr(default_model_arg, 'video_tower', None) is not None:
        lora_cfg_pretrained.image_size = default_data_args.image_size

    model = model.from_pretrained(
        model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs
    )

    # HF occasionally republishes the same base checkpoint with extra
    # placeholder embeddings. Shrink the token embeddings back to the
    # tokenizer vocab before we attach the LoRA weights so that PEFT
    # does not try to copy 151k-row tensors into a 152k-row module.
    target_vocab = len(tokenizer)
    input_embeddings = model.get_input_embeddings()
    if input_embeddings is not None and input_embeddings.weight.size(0) != target_vocab:
        model.resize_token_embeddings(target_vocab)

    if getattr(default_model_arg, 'video_tower', None) is not None:
        model.get_model().initialize_vision_modules(
            model_args=lora_cfg_pretrained,
            fsdp=None,
        )

    from peft import PeftModel

    print('Loading LoRA weights...')
    model = PeftModel.from_pretrained(model, model_path)
    print('Merging LoRA weights...')
    model = model.merge_and_unload()
    print('Model is loaded...')

    target_device = torch.device("cuda") if device == "cuda" else torch.device(device)
    model.to(target_device)
    vt = model.get_vision_tower()
    if vt is not None and hasattr(vt, 'vision_tower'):
        vt.vision_tower.to(target_device)

    model.initialize_vision_tokenizer(lora_cfg_pretrained, tokenizer=tokenizer)

    if hasattr(model, 'initialize_custom_config'):
        model.initialize_custom_config(lora_cfg_pretrained)
        model.to(target_device)
        model_dtype = next(model.parameters()).dtype
        if hasattr(model, 'input_mapping') and isinstance(model.input_mapping, torch.nn.Module):
            model.input_mapping.to(device=target_device, dtype=model_dtype)

    non_lora_path = os.path.join(model_path, 'non_lora_trainables.bin')
    if os.path.exists(non_lora_path):
        obj = torch.load(non_lora_path, map_location='cpu')
        src = obj.get('state_dict', obj) if isinstance(obj, dict) else obj
        if not isinstance(src, dict):
            print('non_lora_trainables is not a dict-like state_dict; skipping load.')
            src = {}
        tgt_state = model.state_dict()
        print(f"Found {len(src)} non-LoRA tensors in checkpoint.")

        def normalize_key(key: str) -> str:
            for prefix in (
                'peft_model.base_model.model.',
                'peft_module.base_model.model.',
                'base_model.model.',
                'model.',
            ):
                if key.startswith(prefix):
                    return key[len(prefix):]
            return key

        remapped = {}
        failed_original = []
        for key, value in src.items():
            norm_key = normalize_key(key)
            if norm_key in tgt_state and hasattr(value, 'shape') and tgt_state[norm_key].shape == value.shape:
                remapped[norm_key] = value
            else:
                failed_original.append(key)

        if remapped:
            _, unexpected = model.load_state_dict(remapped, strict=False)
            print(f"Loaded {len(remapped)} non-LoRA weights.")
            if unexpected:
                print("unexpected non-LoRA weights:", unexpected)
        else:
            print('No compatible non-LoRA tensors found to load.')
        if failed_original:
            print("Unloaded keys:", failed_original)
    else:
        print('No non-lora trainables found, skipping additional weight load.')
    
    image_processor = None
    vision_tower = model.get_vision_tower()
    if vision_tower is not None:
        image_processor = vision_tower.image_processor

    if hasattr(model.config, 'max_sequence_length'):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048
    
    return tokenizer, model, image_processor, context_len

def load_the_videovae_feature(path):
    try:
        fast_feat = torch.load(path).squeeze(dim=0)  # torch.Size([1, 4, 20, 28, 28]) ->  torch.Size([4, 20, 28, 28]) (C, T, H, W)
        # print('fast_feat:', fast_feat.shape)
    except:
        import ipdb
        ipdb.set_trace()        
    # further downsample the feature
    if fast_feat.shape[1] > 6:
        fast_feat, frame_num, final_fps = fps_base_temporal_sampling(fast_feat.permute([1, 0, 2, 3]), 
                                                                    24, 
                                                                    4, 
                                                                    min_frame_num=0) # (C, T, H, W) -> (T, C, H, W) -> (C, T, H, W)
        fast_feat = fast_feat.permute([1, 0, 2, 3])
    fast_feat = fast_feat.half().cuda()
    
    return fast_feat, frame_num, final_fps


@register_model("unimvu_uni")
@register_model("unimvu")
class UniMVU(lmms):
    """
    UniMVU evaluation model built on the LLaVA-style runtime.
    """

    def __init__(
        self,
        model_path: str = "unimvuv/checkpoint",      # the checkpoint path of the current version
        model_base: str = None,                       # the path of the base model
        model_arg_name: str = None,                   # model arg name
        fast_feat_type: str = 'dense_video',
        slow_feat_type: str = 'siglip',
        truncation: Optional[bool] = True,
        device: Optional[str] = "cuda:0",
        batch_size: Optional[Union[int, str]] = 1,
        model_name: Optional[str] = None,
        attn_implementation: Optional[str] = best_fit_attn_implementation,
        device_map: Optional[str] = "cuda:0",
        conv_template: Optional[str] = "vicuna_v1",
        use_cache: Optional[bool] = True,
        truncate_context: Optional[bool] = False,  # whether to truncate the context in generation, set it False for LLaVA-1.6
        customized_config: Optional[str] = None,  # ends in json
        max_frames_num: Optional[int] = 32,
        mm_spatial_pool_stride: Optional[int] = 2,
        mm_spatial_pool_mode: Optional[str] = "bilinear",
        token_strategy: Optional[str] = "single",  # could be "single" or "multiple", "multiple" denotes adding multiple <image> tokens for each frame
        video_decode_backend: str = "decord",
        video_decode_device: str = "cpu",
        **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._device = torch.device(device)
            self.device_map = device_map
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"

        llava_model_args = {
            "multimodal": True,
        }
        if customized_config is not None:
            llava_model_args["customized_config"] = customized_config
        if attn_implementation is not None:
            llava_model_args["attn_implementation"] = attn_implementation
        if "use_flash_attention_2" in kwargs:
            llava_model_args["use_flash_attention_2"] = kwargs["use_flash_attention_2"]
        model_name = model_name if model_name is not None else get_model_name_from_path(model_path)
        self.model_name = model_name

        self.model_path = model_path
        self.token_strategy = token_strategy
        self.max_frames_num = max_frames_num
        self.mm_spatial_pool_stride = mm_spatial_pool_stride
        self.mm_spatial_pool_mode = mm_spatial_pool_mode
        self.video_decode_backend = video_decode_backend
        if video_decode_device == "auto":
            if self._device.type == "cuda":
                device_index = (
                    self._device.index
                    if self._device.index is not None
                    else torch.cuda.current_device()
                )
                self.video_decode_device = decord.gpu(device_index)
            else:
                self.video_decode_device = decord.cpu(0)
        else:
            self.video_decode_device = decord.cpu(0)
        self.fast_feat_type = fast_feat_type
        self.slow_feat_type = slow_feat_type

        overwrite_config = {}
        overwrite_config["mm_spatial_pool_stride"] = self.mm_spatial_pool_stride
        overwrite_config["mm_spatial_pool_mode"] = self.mm_spatial_pool_mode
        # cfg_pretrained = AutoConfig.from_pretrained(self.model_path)

        llava_model_args["overwrite_config"] = overwrite_config
        # try:
        #     # Try to load the model with the multimodal argument
        #     self._tokenizer, self._model, self._image_processor, self._max_length = load_pretrained_model(pretrained, None, model_name, device_map=self.device_map, **llava_model_args)
        # except TypeError:
        #     # for older versions of LLaVA that don't have multimodal argument
        #     llava_model_args.pop("multimodal", None)
        #     self._tokenizer, self._model, self._image_processor, self._max_length = load_pretrained_model(pretrained, None, model_name, device_map=self.device_map, **llava_model_args)
        
        if model_arg_name is None:
            model_arg_name = 'VideoFeatModelArgumentsUniMVU'

        # load the model, return the model, tokenize, image_processor, max_length
        self._tokenizer, self._model, self._image_processor, self._max_length = load_trained_model_for_eval_unimvuv(
            model_path,
            model_base,
            model_name,
            model_arg_name=model_arg_name,
        )
        
        # update the config?
        # import ipdb
        # ipdb.set_trace() # check the loading result, check the config and other?
        

        self._config = self._model.config
        self.model.eval()
        self.truncation = truncation
        self.batch_size_per_gpu = int(batch_size)
        self.conv_template = conv_template
        self.use_cache = use_cache
        self.truncate_context = truncate_context
        assert self.batch_size_per_gpu == 1, "Llava currently does not support batched generation. See https://github.com/haotian-liu/LLaVA/issues/754. HF Llava also has this issue."

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            # If you want to use DistributedType.DEEPSPEED, you have to run accelerate config before using the model
            # Also, you have to select zero stage 0 (equivalent to DDP) in order to make the prepare model works
            # I tried to set different parameters in the kwargs to let default zero 2 stage works, but it didn't work.
            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs = {
                    "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,
                    "train_batch_size": self.batch_size_per_gpu * accelerator.num_processes,
                }
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(must_match=True, **kwargs)
                eval_logger.info("Detected that you are using DistributedType.DEEPSPEED. Make sure you run `accelerate config` and set zero stage to 0")

            if accelerator.distributed_type == DistributedType.FSDP or accelerator.distributed_type == DistributedType.DEEPSPEED:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes

        elif accelerator.num_processes == 1 and device_map == "auto":
            eval_logger.info(f"Using {accelerator.num_processes} devices with tensor parallelism")
            self._rank = 0
            self._world_size = 1

        else:
            eval_logger.info(f"Using single device: {self._device}")
            self.model.to(self._device)
            self._rank = 0
            self._world_size = 1

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def tok_encode(self, string: str, left_truncate_len=None, add_special_tokens=None) -> List[int]:
        """ """
        add_special_tokens = False if add_special_tokens is None else add_special_tokens
        encoding = self.tokenizer.encode(string, add_special_tokens=add_special_tokens)
        # left-truncate the encoded context to be at most `left_truncate_len` tokens long
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]
        return encoding

    def tok_decode(self, tokens):
        try:
            return self.tokenizer.decode(tokens)
        except:
            return self.tokenizer.decode([tokens])

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def load_video(self, video_path, max_frames_num):
        if os.path.isdir(video_path):
            valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
            frame_files = [
                fname for fname in os.listdir(video_path) if fname.lower().endswith(valid_exts)
            ]
            if not frame_files:
                raise FileNotFoundError(f"No frame images found in directory: {video_path}")
            frame_files.sort(key=_natural_key)
            indices = np.linspace(0, len(frame_files) - 1, max_frames_num, dtype=int)
            frames = []
            for idx in indices:
                frame_path = os.path.join(video_path, frame_files[idx])
                with PIL.Image.open(frame_path) as img:
                    frames.append(np.array(img.convert("RGB")))
            return np.stack(frames, axis=0)

        vr = SafeVideoReader(video_path, ctx=self.video_decode_device)
        total_frame_num = len(vr)
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        spare_frames = vr.get_batch(frame_idx).numpy()
        return spare_frames  # (frames, height, width, channels)

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tok_encode(x[0])
            return -len(toks), x[0]

        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        metadata = requests[0].metadata
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")
        for chunk in chunks:
            batched_contexts, all_gen_kwargs, batched_doc_to_visual, batched_doc_id, batched_task, batched_split = zip(*chunk)
            task = batched_task[0]
            split = batched_split[0]
            # import ipdb
            # ipdb.set_trace() # check the following function call
            # temp = self.task_dict['activitynetqa']['test'][0]
            # temp = batched_doc_to_visual[0](temp)
            batched_visuals = [batched_doc_to_visual[0](self.task_dict[task][split][ids]) for ids in batched_doc_id]  # [B, N]
            # ipdb> self.task_dict.keys()
            # dict_keys(['activitynetqa'])
            # ipdb> self.task_dict['activitynetqa'].keys()
            # dict_keys(['test'])
            # ipdb> self.task_dict['activitynetqa']['test'].keys()
            # *** AttributeError: 'Dataset' object has no attribute 'keys'
            # ipdb> self.task_dict['activitynetqa']['test'][0]
            # {'video_name': '1QIUV7WYKXg', 'question_id': 'v_1QIUV7WYKXg_3', 'question': 'is the athlete wearing trousers', 'answer': 'no', 'type': '3'}
            # ipdb>
            
            
            assert len(batched_visuals) == 1

            # we assume all gen kwargs in the batch are the same
            # this is safe to assume because the `grouper` object ensures it.
            gen_kwargs = all_gen_kwargs[0]
            if "until" in gen_kwargs:
                gen_kwargs.pop("until")

            question_input = []
            # import ipdb
            # ipdb.set_trace()
            for visual, context in zip(batched_visuals, batched_contexts):
                if visual is None or visual == []:
                    # Handle text-only evaluation - set up dummy values for model.generate
                    visual = None
                    task_type = "text"
                    placeholder_count = 0
                    image_tensor = None
                    # Initialize dummy values for video features (model.generate requires these)
                    fast_feat = torch.zeros((1, 1, 1, 1), dtype=torch.float16).cuda()
                    frame_num = 1
                    final_fps = 1
                    image_sizes = -1
                elif len(visual) == 1:
                    original_path = visual[0]
                    slow_cache_dir = os.environ.get("UNIMVU_MVBENCH_VIDEO_ROOT")
                    fast_cache_dir = os.environ.get("UNIMVU_MVBENCH_FEATURE_ROOT")

                    if not slow_cache_dir or not fast_cache_dir:
                        raise ValueError(
                            "MVBench evaluation requires UNIMVU_MVBENCH_VIDEO_ROOT and "
                            "UNIMVU_MVBENCH_FEATURE_ROOT to be set."
                        )

                    if 'mvbench' in original_path:
                        relative_path = original_path.split('mvbench', 1)[-1].lstrip(os.sep)
                    elif 'MVBench' in original_path:
                        relative_path = original_path.split('MVBench', 1)[-1].lstrip(os.sep)
                    else:
                        relative_path = os.path.basename(original_path)

                    base_path, ext = os.path.splitext(relative_path)
                    if ext.lower() in {'.mp4', '.mkv', '.webm', '.avi', '.mov'}:
                        feature_rel_path = base_path
                    else:
                        feature_rel_path = relative_path

                    slow_path = os.path.join(slow_cache_dir, feature_rel_path) + '.pt'
                    fast_path = os.path.join(fast_cache_dir, feature_rel_path) + '.pt'

                    candidate_video_path = os.path.join(slow_cache_dir, relative_path)
                    if os.path.exists(candidate_video_path):
                        video_source = candidate_video_path
                    else:
                        video_source = original_path

                    visual = [slow_path, fast_path, video_source]
                
                if visual is None or visual == []:  # for text-only tasks.
                    visual = None
                    task_type = "text"
                    placeholder_count = 0
                    image_tensor = None
                    # Initialize dummy values for video features (model.generate requires these)
                    fast_feat = torch.zeros((1, 1, 1, 1), dtype=torch.float16).cuda()
                    frame_num = 1
                    final_fps = 1
                    image_sizes = -1
                else:
                    image_tensor = []
                    # visual [slow_path, fast_path]
                    # load the slow
                    if self.slow_feat_type == 'siglip':
                        slow_feat = torch.load(visual[0]).squeeze(dim=0).half().cuda()
                        image_tensor.append(slow_feat)
                        image_sizes = -1             
                    elif self.slow_feat_type == 'raw_video':
                        video_file_name = visual[2]
                        # import ipdb
                        # ipdb.set_trace() # check the shape
                        # load the video by sampling 32 frames 
                        try:
                            if os.path.isdir(video_file_name):
                                frames = self.load_video(video_file_name, self.max_frames_num)
                            elif self.video_decode_backend == "decord":
                                frames = self.load_video(video_file_name, self.max_frames_num)
                            elif self.video_decode_backend == "pyav":
                                frames = read_video_pyav(video_file_name, num_frm=self.max_frames_num)
                            frames = self._image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].half().cuda() # torch.Size([32, 3, 384, 384])
                            image_tensor.append(frames)
                            image_sizes = 100 # TODO: avoid this simple walk around
                        except Exception as e:
                            eval_logger.error(f"Error {e} in loading video")
                            print(f"Error {e} in loading video")
                            image_tensor = None
                            image_sizes = -1
                    else:
                        raise NotImplementedError
                               
                    
                    # load the fast
                    if self.fast_feat_type == 'video_vae':
                        fast_feat, frame_num, final_fps = load_the_videovae_feature(visual[1])
                    elif self.fast_feat_type == 'languagebind' or \
                         self.fast_feat_type == 'dense_video' or \
                         self.fast_feat_type == 'languagebind_14x14' or \
                         self.fast_feat_type == 'internvideo2' or \
                         self.fast_feat_type == 'siglip' or  \
                         self.fast_feat_type == 'audio':
                        # import ipdb
                        # ipdb.set_trace() # check the loading
                        # replace the path
                        if self.fast_feat_type == 'languagebind':
                            path = visual[1].replace('24fps_feat', 'languagebind_feat')
                        elif self.fast_feat_type == 'dense_video':
                            path = visual[1].replace('24fps_feat', 'languagebind_feat')
                        elif self.fast_feat_type == 'languagebind_14x14':
                            path = visual[1].replace('24fps_feat', 'languagebind_feat_14x14')
                        elif self.fast_feat_type == 'internvideo2':
                            path = visual[1].replace('24fps_feat', 'internvideo2_feat')
                        elif self.fast_feat_type == 'siglip':
                            path = visual[1].replace('24fps_feat', 'siglip_fast_feat')
                        elif self.fast_feat_type == 'audio':
                            path = visual[1].replace('24fps_feat', 'audio_feat')
                        else:
                            raise NotImplementedError
                        
                        # load the feature 
                        if self.fast_feat_type == 'audio':
                            video_feat = torch.load(path, map_location=torch.device('cpu')) #torch.Size([19474, 768]) T, C
                            video_feat = video_feat.permute([1,0]) # (C, T)
                            # unsqueeze dim 
                            fast_feat = video_feat.unsqueeze(dim=-1).unsqueeze(dim=-1) # (C, T, 1, 1) (C, T, H, W)
                            fast_feat = fast_feat.half().cuda()
                            final_fps = 100 / 16
                            frame_num = fast_feat.shape[1]                          
                        else:
                            if torch.cuda.is_available():
                                map_location = torch.device('cuda')
                            else:
                                map_location = torch.device('cpu')
                            video_feat = torch.load(path, map_location=map_location) # torch.Size([280, 5, 1024]) T, C, D
                            # exclude the cls tokens
                            if self.fast_feat_type == 'languagebind':
                                video_feat = video_feat[:, 1:,]
                                # reshape
                                S = video_feat.shape[1]
                                assert int(math.sqrt(S)) ** 2 == S # assert is a square
                                W = H = int(math.sqrt(S))
                                fast_feat = rearrange(video_feat, 't (h w) c -> c t h w', h = H) # video_feat should be in the shape of (C, T, H, W)
                                fast_feat = fast_feat.half().cuda()
                                final_fps = 2
                                frame_num = fast_feat.shape[1]  
                            if self.fast_feat_type == 'dense_video':
                                # print('fast_feat:', fast_feat.shape)
                                # We use cls token here, so the shape of video_feat is (T, D, C) # D = 5, C = 1024
                                # if 'MVBench' in visual[0]:
                                #     fast_feat = video_feat.cuda()
                                #     # fast_feat = fast_feat[:, 1:,]
                                #     # S = fast_feat.shape[1]
                                #     # assert int(math.sqrt(S)) ** 2 == S # assert is a square
                                #     # W = H = int(math.sqrt(S))
                                #     # fast_feat = rearrange(fast_feat, 't (h w) c -> t h w c', h = H, w = W)
                                #     frame_num = fast_feat.shape[0]
                                #     final_fps = 1

                                #Remove CLS token version
                                if 'MVBench' in visual[0]:
                                    fast_feat = video_feat.cuda()
                                    fast_feat = fast_feat[:, 1:,]
                                    # S = fast_feat.shape[1]
                                    # assert int(math.sqrt(S)) ** 2 == S # assert is a square
                                    # W = H = int(math.sqrt(S))
                                    # fast_feat = rearrange(fast_feat, 't (h w) c -> t h w c', h = H, w = W)
                                    frame_num = fast_feat.shape[0]
                                    final_fps = 1                                
                                else:
                                    video_feat = video_feat[:, 1:,]
                                    # reshape
                                    S = video_feat.shape[1]
                                    assert int(math.sqrt(S)) ** 2 == S # assert is a square
                                    W = H = int(math.sqrt(S))
                                    fast_feat = video_feat.half().cuda()
                                    fast_feat = rearrange(fast_feat, 't (h w) c -> t h w c', h = H, w = W)
                                    # Target 2fps to align with training data (Mean=102, Max=352)
                                    # Use min_frame_num=1 to avoid artificial FPS inflation for short videos
                                    fast_feat, frame_num, final_fps = fps_base_temporal_sampling(fast_feat, 
                                                                24, 
                                                                2, 
                                                                min_frame_num=1)
                                    
                                    # Cap at 320 frames (approx Training P95-P99)
                                    # This allows full utilization of trained positional embeddings without extreme OOD
                                    # while covering long videos by reducing effective sampling rate
                                    max_frame_num = 320
                                    if frame_num > max_frame_num:
                                        # Uniformly select max_frame_num frames from the available frames
                                        # Deterministic sampling is better for evaluation
                                        selected_indices = torch.linspace(0, frame_num - 1, max_frame_num).long().to(fast_feat.device)
                                        fast_feat = fast_feat[selected_indices]
                                        
                                        # Update FPS to reflect the new number of frames over the same duration
                                        final_fps = final_fps * (max_frame_num / frame_num)
                                        frame_num = max_frame_num
                  
                    else:
                        raise NotImplementedError

                    task_type = "video"
                    placeholder_count = len(slow_feat) if self.token_strategy == "multiple" else 1

                if image_tensor is not None and len(image_tensor) != 0 and DEFAULT_IMAGE_TOKEN not in context:
                    """
                    Three senarios:
                    1. No image, and there for, no image token should be added.
                    2. image token is already specified in the context, so we don't need to add it.
                    3. image token is not specified in the context and there is image inputs, so we need to add it. In this case, we add the image token at the beginning of the context and add a new line.
                    4. For video tasks, we could add a <image> token or multiple <image> tokens for each frame in the context. This depends on the training strategy and should balance in test to decide which is better
                    """
                    # if task_type == "image": # indeed in multi-image case, not the video in frames.
                    #     image_tokens = [DEFAULT_IMAGE_TOKEN] * placeholder_count if isinstance(visual, list) else [DEFAULT_IMAGE_TOKEN]
                    # elif task_type == "video":
                    # image_tokens = [DEFAULT_IMAGE_TOKEN] * placeholder_count if self.token_strategy == "multiple" else [DEFAULT_IMAGE_TOKEN]
                    image_tokens = [DEFAULT_IMAGE_TOKEN] * placeholder_count
                    image_tokens = " ".join(image_tokens)
                    question = image_tokens + "\n" + context
                else:
                    question = context

                # This is much safer for llama3, as we now have some object type in it
                if "llama_3" in self.conv_template:
                    conv = copy.deepcopy(conv_templates[self.conv_template])
                else:
                    conv = conv_templates[self.conv_template].copy()

                if utils.is_json(question):  # conversational question input
                    question = json.loads(question)
                    for idx, item in enumerate(question):
                        role = conv.roles[idx % 2]
                        message = item["value"]
                        conv.append_message(role, message)

                    assert len(conv.messages) % 2 == 1
                    conv.append_message(conv.roles[1], None)
                    prompt_question = conv.get_prompt()
                    question_input.append(prompt_question)
                else:  # only simple string for question
                    conv.append_message(conv.roles[0], question)
                    conv.append_message(conv.roles[1], None)
                    prompt_question = conv.get_prompt()
                    question_input.append(prompt_question)

            # preconfigure gen_kwargs with defaults
            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "do_sample" not in gen_kwargs:
                gen_kwargs["do_sample"] = False
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1

            input_ids_list = [tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt") for prompt in question_input]
            pad_token_ids = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            input_ids = self.pad_sequence(input_ids_list, batch_first=True, padding_value=pad_token_ids).to(self.device)
            attention_masks = input_ids.ne(pad_token_ids).to(self.device)

            if task_type == "image":
                raise NotImplementedError
                # gen_kwargs["image_sizes"] = [batched_visuals[0][idx].size for idx in range(len(batched_visuals[0]))]
            elif task_type == "video":
                stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                keywords = [stop_str]
                stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
                gen_kwargs["modalities"] = ["video"] + [self.fast_feat_type]
                gen_kwargs["stopping_criteria"] = [stopping_criteria]
                self._config.mm_spatial_pool_stride = self.mm_spatial_pool_stride
                self._config.mm_spatial_pool_mode = self.mm_spatial_pool_mode
                fast_feat = fast_feat.unsqueeze(dim=0)
            if image_tensor is None:
                res.extend(["Error in loading video"])
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), ["Error in loading video"])
                pbar.update(1)
                continue
            elif task_type == "text":
                fast_feat = None
            # These steps are not in LLaVA's original code, but are necessary for generation to work
            # TODO: attention to this major generation step...
            if "image_aspect_ratio" in gen_kwargs.keys():
                gen_kwargs.pop("image_aspect_ratio")
            try:
                with torch.inference_mode():
                    cont = self.model.generate(input_ids, 
                                               video_feats=fast_feat,
                                               video_feat_fps=torch.tensor([final_fps]).cuda(),
                                               feat_frame_nums=torch.tensor([frame_num]).cuda(),
                                               images=image_tensor,
                                               image_sizes=[image_sizes], # -1 as an indicator of using the slow feature
                                               
                                               attention_mask=attention_masks, 
                                               pad_token_id=pad_token_ids, 
                                               use_cache=self.use_cache, 
                                               cache_position=None,
                                               **gen_kwargs)

                text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)
                
            except Exception as e:
                raise e

            text_outputs = [response.strip() for response in text_outputs]
            res.extend(text_outputs)
            self.cache_hook.add_partial("generate_until", (context, gen_kwargs), text_outputs)
            pbar.update(1)
            # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)

        pbar.close()
        return res

    def generate_until_multi_round(self, requests: List[Instance]) -> List[str]:
        raise NotImplementedError
