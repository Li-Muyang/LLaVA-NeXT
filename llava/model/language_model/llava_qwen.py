#    Copyright 2024 Hao Zhang
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

import transformers
from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

# from ...constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from transformers import Qwen2Config, Qwen2Model, Qwen2ForCausalLM


def compute_relational_loss(visual_features: List[torch.Tensor], hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Compute relational alignment loss between projected visual features and LLM hidden states.
    
    Args:
        visual_features: List of projected visual features, one tensor per sample [num_visual_tokens, hidden_dim]
        hidden_states: LLM second-to-last layer hidden states [batch, seq_len, hidden_dim]
        attention_mask: Attention mask [batch, seq_len]
    
    Returns:
        MSE loss between pairwise cosine distance matrices
    """
    batch_size = len(visual_features)
    if batch_size < 2:
        return torch.tensor(0.0, device=hidden_states.device, dtype=hidden_states.dtype)
    
    # Mean pool visual features per sample
    visual_pooled = []
    for vf in visual_features:
        if vf.dim() == 2:
            visual_pooled.append(vf.mean(dim=0))  # [hidden_dim]
        else:
            visual_pooled.append(vf.flatten(0, -2).mean(dim=0))  # Handle any shape
    visual_pooled = torch.stack(visual_pooled)  # [batch, hidden_dim]
    
    # Mean pool hidden states per sample (masked)
    # hidden_states: [batch, seq_len, hidden_dim]
    mask = attention_mask.unsqueeze(-1).float()  # [batch, seq_len, 1]
    hidden_pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # [batch, hidden_dim]
    
    # Normalize for cosine distance
    visual_norm = F.normalize(visual_pooled, p=2, dim=-1)
    hidden_norm = F.normalize(hidden_pooled, p=2, dim=-1)
    
    # Compute pairwise cosine similarity matrices
    visual_sim = torch.mm(visual_norm, visual_norm.t())  # [batch, batch]
    hidden_sim = torch.mm(hidden_norm, hidden_norm.t())  # [batch, batch]
    
    # Cosine distance = 1 - cosine similarity
    visual_dist = 1 - visual_sim
    hidden_dist = 1 - hidden_sim
    
    # MSE between distance matrices
    loss = F.mse_loss(visual_dist, hidden_dist)
    
    return loss

# from .qwen.modeling_qwen import QWenLMHeadModel, QWenModel
# from .qwen.configuration_qwen import QWenConfig


class LlavaQwenConfig(Qwen2Config):
    model_type = "llava_qwen"


class LlavaQwenModel(LlavaMetaModel, Qwen2Model):
    config_class = LlavaQwenConfig

    def __init__(self, config: Qwen2Config):
        super(LlavaQwenModel, self).__init__(config)


class LlavaQwenForCausalLM(Qwen2ForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaQwenConfig

    def __init__(self, config):
        # super(Qwen2ForCausalLM, self).__init__(config)
        Qwen2ForCausalLM.__init__(self, config)
        config.model_type = "llava_qwen"
        config.rope_scaling = None

        self.model = LlavaQwenModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def _is_stage1_training(self) -> bool:
        """Check if we are in Stage 1 (pretraining/alignment) based on config."""
        # Stage 1: version == "plain" or only training mm_mlp_adapter
        mm_tunable_parts = getattr(self.config, 'mm_tunable_parts', None)
        tune_mm_mlp_adapter = getattr(self.config, 'tune_mm_mlp_adapter', False)
        
        if mm_tunable_parts == 'mm_mlp_adapter':
            return True
        if tune_mm_mlp_adapter and not getattr(self.config, 'unfreeze_mm_vision_tower', False):
            return True
        return False

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        modalities: Optional[List[str]] = ["image"],
        dpo_forward: Optional[bool] = False,
        cache_position=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        # Check if relational loss should be computed
        relational_loss_weight = getattr(self.config, 'relational_loss_weight', 0.0)
        compute_relational = (
            self.training and 
            relational_loss_weight > 0 and 
            images is not None and 
            self._is_stage1_training()
        )
        
        # Store projected visual features if computing relational loss
        projected_visual_features = None
        if compute_relational and images is not None:
            # Encode images to get projected features before prepare_inputs_labels_for_multimodal
            # We need to manually encode to capture the projected features
            if type(images) is list or images.ndim == 5:
                if type(images) is list:
                    images_list = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
                else:
                    images_list = [images]
                concat_images = torch.cat([img for img in images_list], dim=0)
                split_sizes = [img.shape[0] for img in images_list]
                # Get projected features
                image_features = self.get_model().get_vision_tower()(concat_images)
                projected_visual_features = self.get_model().mm_projector(image_features)
                # Split back to per-sample
                projected_visual_features = list(torch.split(projected_visual_features, split_sizes, dim=0))
            else:
                image_features = self.get_model().get_vision_tower()(images)
                projected_visual_features = self.get_model().mm_projector(image_features)
                projected_visual_features = [projected_visual_features]

        if inputs_embeds is None:
            (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels) = self.prepare_inputs_labels_for_multimodal(input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities, image_sizes)

        if dpo_forward:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            hidden_states = outputs[0]
            logits = self.lm_head(hidden_states)
            return logits, labels

        # For relational loss, we need hidden states
        if compute_relational:
            output_hidden_states = True

        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,  # Need dict to access hidden_states
        )

        # Compute and add relational loss
        if compute_relational and projected_visual_features is not None and outputs.hidden_states is not None:
            # Get second-to-last layer hidden states
            second_to_last_hidden = outputs.hidden_states[-2]  # [batch, seq_len, hidden_dim]
            
            rel_loss = compute_relational_loss(
                projected_visual_features, 
                second_to_last_hidden, 
                attention_mask
            )
            
            # Add weighted relational loss to total loss
            if outputs.loss is not None:
                outputs.loss = outputs.loss + relational_loss_weight * rel_loss

        # Convert back to tuple if return_dict is False
        if return_dict is False:
            return outputs.to_tuple()
        
        return outputs

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        modalities: Optional[List[str]] = ["image"],
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (inputs, position_ids, attention_mask, _, inputs_embeds, _) = self.prepare_inputs_labels_for_multimodal(inputs, position_ids, attention_mask, None, None, images, modalities, image_sizes=image_sizes)
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(position_ids=position_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs)
        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        return inputs


AutoConfig.register("llava_qwen", LlavaQwenConfig)
AutoModelForCausalLM.register(LlavaQwenConfig, LlavaQwenForCausalLM)
