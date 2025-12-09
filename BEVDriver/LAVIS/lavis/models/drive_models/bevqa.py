import logging
import re
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from lavis.common.registry import registry
from lavis.models.blip2_models.blip2 import disabled_train
from lavis.models.bevllm_models.bevllm import BEVLLMBase as Blip2Base

from timm import create_model


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


@registry.register_model("bevqa")
class BEVQAModel(Blip2Base):
    """
    BEV-QA model: Reuses BEV encoder + Q-Former to produce visual tokens
    and conditions a causal LLM to generate VQA answers.

    Inputs (samples dict expected keys):
      - rgb, rgb_left, rgb_right, rgb_center: images (B[, T], C, H, W)
      - lidar: point cloud/raster input as used by bevdriver_encoder
      - measurements: driving metadata (mask, velocity, etc.)
      - vqa_question: list[str] question texts
      - vqa_answer: list[str] answer texts (for training)
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "base": "configs/models/bevqa.yaml",
    }

    def __init__(
        self,
        *,
        encoder_model: str,
        encoder_model_ckpt: str = "",
        load_pretrained: bool = True,
        freeze_vit: bool = True,
        llm_model: str,
        max_txt_len: int = 128,
        num_query_token: int = 32,
        has_lora: bool = True,
        first_sentence_only: bool = False,
    ):
        super().__init__()

        # LLM imports kept local to avoid global overhead
        from transformers import LlamaTokenizer, GenerationConfig
        from lavis.models.blip2_models.modeling_llama import LlamaForCausalLM
        from lavis.models.blip2_models.modeling_opt import OPTForCausalLM
        from transformers import AutoTokenizer

        self.tokenizer = self.init_tokenizer(truncation_side="left")

        # BEV encoder (timm registry: bevdriver_encoder)
        self.bev_encoder = create_model(encoder_model)
        logging.info(f"BEV encoder embed dim: {self.bev_encoder.embed_dim}")
        self.ln_vision = LayerNorm(self.bev_encoder.embed_dim)

        if load_pretrained and encoder_model_ckpt:
            pretrain_weights = torch.load(encoder_model_ckpt, map_location=torch.device("cpu"))
            # support both full and state_dict-only checkpoints
            state = pretrain_weights.get("state_dict", pretrain_weights)
            self.bev_encoder.load_state_dict(state, strict=False)

        if freeze_vit:
            for name, param in self.bev_encoder.named_parameters():
                param.requires_grad = False
            self.bev_encoder = self.bev_encoder.eval()
            self.bev_encoder.train = disabled_train

        # LLM backbone (OPT or LLaMA family)
        self.is_opt = "opt" in llm_model.lower()
        if self.is_opt:
            self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_model, use_fast=False, truncation_side="left")
            self.llm_model = OPTForCausalLM.from_pretrained(
                llm_model, torch_dtype=torch.float16, low_cpu_mem_usage=True
            )
        else:
            self.llm_tokenizer = LlamaTokenizer.from_pretrained(llm_model, use_fast=False, truncation_side="left")
            self.llm_model = LlamaForCausalLM.from_pretrained(
                llm_model, torch_dtype=torch.float16, low_cpu_mem_usage=True
            )

        # Ensure generation_config exists for newer transformers versions.
        if getattr(self.llm_model, "generation_config", None) is None:
            self.llm_model.generation_config = GenerationConfig.from_model_config(self.llm_model.config)

        # Disable generation cache to avoid incompatibilities with custom LLaMA implementation
        # and simplify generation behavior across transformers versions.
        if hasattr(self.llm_model, "config"):
            self.llm_model.config.use_cache = False
        if getattr(self.llm_model, "generation_config", None) is not None:
            self.llm_model.generation_config.use_cache = False

        # Ensure special tokens exist and resize embeddings
        self.llm_tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        # For some HF checkpoints, reuse </s> as several specials
        self.llm_tokenizer.add_special_tokens({"bos_token": "</s>", "eos_token": "</s>", "unk_token": "</s>"})
        self.llm_model.resize_token_embeddings(len(self.llm_tokenizer))

        # Optional LoRA fine-tuning
        self.has_lora = has_lora
        self.first_sentence_only = first_sentence_only
        if has_lora:
            from peft import LoraConfig, get_peft_model

            lora_cfg = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.llm_model = get_peft_model(self.llm_model, lora_cfg)
            self.llm_model.print_trainable_parameters()
        else:
            for _, p in self.llm_model.named_parameters():
                p.requires_grad = False

        # Q-Former and projection to LLM hidden size
        self.Qformer, self.query_tokens = self.init_Qformer(num_query_token, self.bev_encoder.embed_dim)
        # tie tokenizer for Q-Former with LLM tokenizer to keep vocab consistent
        self.Qformer.resize_token_embeddings(len(self.llm_tokenizer))
        self.Qformer.cls = None

        self.llm_proj = nn.Linear(self.Qformer.config.hidden_size, self.llm_model.config.hidden_size)
        self.max_txt_len = max_txt_len

    def get_optimizer_params(self, weight_decay, lr_scale=1):
        """
        Override optimizer grouping to avoid referencing visual_encoder.
        Groups parameters by weight decay / no decay similar to BaseModel default.
        """
        decay, no_decay = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim < 2 or name.endswith(".bias") or "ln" in name or "bn" in name:
                no_decay.append(param)
            else:
                decay.append(param)
        optim_params = []
        if decay:
            optim_params.append({"params": decay, "weight_decay": weight_decay, "lr_scale": lr_scale})
        if no_decay:
            optim_params.append({"params": no_decay, "weight_decay": 0.0, "lr_scale": lr_scale})
        return optim_params

    def _encode_bev(self, samples: dict) -> torch.Tensor:
        """Runs BEV encoder and returns token sequence [B, S, D].
        Supports inputs with optional time dimension by flattening B*T.
        """
        rgb = samples["rgb"]
        bt = rgb.shape[0]
        # If inputs are shaped [B, T, ...], flatten to [B*T, ...]
        if rgb.dim() == 5:
            bs, t = rgb.shape[:2]
            def _flat(x):
                shape = x.shape
                return x.view(bs * t, *shape[2:])
            bev_inp = {
                "rgb": _flat(samples["rgb"]),
                "rgb_left": _flat(samples["rgb_left"]),
                "rgb_right": _flat(samples["rgb_right"]),
                "rgb_center": _flat(samples["rgb_center"]),
                "lidar": _flat(samples["lidar"]),
                "measurements": _flat(samples["measurements"]),
            }
        else:
            bev_inp = {
                "rgb": samples["rgb"],
                "rgb_left": samples["rgb_left"],
                "rgb_right": samples["rgb_right"],
                "rgb_center": samples["rgb_center"],
                "lidar": samples["lidar"],
                "measurements": samples["measurements"],
            }

        with self.maybe_autocast():
            memory = self.bev_encoder(bev_inp)  # [N, S, D]
        return self.ln_vision(memory)

    def _qformer_visual_tokens(self, bev_tokens: torch.Tensor) -> torch.Tensor:
        """Compress BEV tokens with Q-Former and project to LLM hidden size.
        Returns [B, Q, H].
        """
        atts = torch.ones(bev_tokens.size()[:-1], dtype=torch.long, device=bev_tokens.device)
        q_tokens = self.query_tokens.expand(bev_tokens.shape[0], -1, -1)
        q_out = self.Qformer.bert(
            query_embeds=q_tokens,
            encoder_hidden_states=bev_tokens,
            encoder_attention_mask=atts,
            use_cache=True,
            return_dict=True,
        )
        vis = self.llm_proj(q_out.last_hidden_state)  # [B, Q, H]
        return vis

    def _build_prompt(self, questions: List[str]) -> List[str]:
        return [
            (
                "Question: {question}\n"
                "Answer in one concise sentence summarizing what you see."
            ).format(question=q)
            for q in questions
        ]

    @staticmethod
    def _first_sentence(text: str) -> str:
        if not isinstance(text, str):
            return text
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        return parts[0] if parts else text

    def forward(self, samples: dict, *, bev_memory: Optional[torch.Tensor] = None):
        """
        Training forward: computes LM loss on answer tokens only.
        Expects samples to contain 'vqa_question' and 'vqa_answer' lists of strings.
        """
        device = samples["rgb"].device
        questions: List[str] = samples["vqa_question"]
        answers: List[str] = samples["vqa_answer"]
        if self.first_sentence_only:
            answers = [self._first_sentence(a) for a in answers]

        if bev_memory is None:
            bev_tokens = self._encode_bev(samples)  # [B, S, D]
        else:
            bev_tokens = bev_memory

        vis_tokens = self._qformer_visual_tokens(bev_tokens)  # [B, Q, H]
        llm_dtype = self.llm_model.get_input_embeddings().weight.dtype
        vis_tokens = vis_tokens.to(llm_dtype)

        prompts = self._build_prompt(questions)

        prompt_tokens = self.llm_tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(device)

        answer_tokens = self.llm_tokenizer(
            answers,
            padding=True,
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(device)

        # Build inputs_embeds = [prompt_embeds, vis_tokens, answer_embeds]
        prompt_embeds = self.llm_model.get_input_embeddings()(prompt_tokens.input_ids).to(llm_dtype)
        answer_embeds = self.llm_model.get_input_embeddings()(answer_tokens.input_ids).to(llm_dtype)

        inputs_embeds = torch.cat([prompt_embeds, vis_tokens, answer_embeds], dim=1)

        attn_prompt = prompt_tokens.attention_mask
        attn_vis = torch.ones(vis_tokens.size()[:2], dtype=torch.long, device=device)
        attn_ans = answer_tokens.attention_mask
        attention_mask = torch.cat([attn_prompt, attn_vis, attn_ans], dim=1)

        # Labels: ignore prompt + visual tokens; supervise only on answer tokens
        ignore = torch.full(attn_prompt.size(), -100, dtype=torch.long, device=device)
        ignore_vis = torch.full(attn_vis.size(), -100, dtype=torch.long, device=device)
        answer_labels = answer_tokens.input_ids.masked_fill(answer_tokens.attention_mask == 0, -100)
        labels = torch.cat([ignore, ignore_vis, answer_labels], dim=1)

        out = self.llm_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )
        if isinstance(out, dict):
            loss = out.get("loss", None)
        else:
            loss = getattr(out, "loss", None)
        if loss is None:
            if isinstance(out, (tuple, list)) and len(out) > 0 and torch.is_tensor(out[0]):
                loss = out[0]
            elif torch.is_tensor(out):
                loss = out
            else:
                raise RuntimeError("LLM forward did not return a loss term.")

        if torch.is_tensor(loss) and loss.dim() > 0:
            loss = loss.mean()

        return {"loss": loss, "lm_loss": loss}

    @torch.no_grad()
    def generate(
        self,
        samples: dict,
        *,
        bev_memory: Optional[torch.Tensor] = None,
        max_new_tokens: int = 32,
        num_beams: int = 1,
        top_p: float = 0.9,
        temperature: float = 1.0,
        repetition_penalty: float = 1.0,
        min_new_tokens: int = 0,
    ) -> List[str]:
        # NOTE: We implement a custom generation loop instead of relying on
        # `llm_model.generate` because some PEFT + custom LLaMA combinations
        # can trigger shape mismatches in HF generation utilities when using
        # `inputs_embeds` (e.g., zero-length query steps). This path trades
        # some efficiency for robustness.
        device = samples["rgb"].device
        questions: List[str] = samples["vqa_question"]

        if bev_memory is None:
            bev_tokens = self._encode_bev(samples)
        else:
            bev_tokens = bev_memory

        vis_tokens = self._qformer_visual_tokens(bev_tokens)  # [B, Q, H]
        llm_dtype = self.llm_model.get_input_embeddings().weight.dtype
        vis_tokens = vis_tokens.to(llm_dtype)

        prompts = self._build_prompt(questions)
        prompt_tokens = self.llm_tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(device)
        prompt_embeds = self.llm_model.get_input_embeddings()(prompt_tokens.input_ids).to(llm_dtype)
        prefix_embeds = torch.cat([prompt_embeds, vis_tokens], dim=1)
        prefix_attn = torch.cat(
            [prompt_tokens.attention_mask, torch.ones(vis_tokens.size()[:2], dtype=torch.long, device=device)],
            dim=1,
        )

        if num_beams != 1:
            raise NotImplementedError("Beam search (num_beams > 1) is not supported in BEVQAModel.generate.")

        batch_size = prefix_embeds.size(0)
        eos_token_id = self.llm_tokenizer.eos_token_id
        pad_token_id = self.llm_tokenizer.pad_token_id
        temperature = max(float(temperature), 1e-4)
        min_new_tokens = max(int(min_new_tokens), 0)

        generated_ids = torch.empty(
            batch_size, 0, dtype=torch.long, device=device
        )
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        neg_inf_value: Optional[float] = None

        def apply_repetition_penalty(logits: torch.Tensor, generated: torch.Tensor, penalty: float):
            if penalty == 1.0 or generated.numel() == 0:
                return logits
            unique_tokens = []
            for row in generated:
                unique_tokens.append(row.tolist())
            # Apply in-place per batch item for efficiency and clarity.
            for b, tokens in enumerate(unique_tokens):
                if not tokens:
                    continue
                tok_counts = set(tokens)
                for tok in tok_counts:
                    val = logits[b, tok]
                    if val > 0:
                        logits[b, tok] = val / penalty
                    else:
                        logits[b, tok] = val * penalty
            return logits

        for step_idx in range(max_new_tokens):
            if generated_ids.size(1) > 0:
                gen_embeds = self.llm_model.get_input_embeddings()(generated_ids)
                inputs_embeds = torch.cat([prefix_embeds, gen_embeds], dim=1)
                attn_prefix = prefix_attn
                attn_gen = torch.ones(
                    batch_size,
                    generated_ids.size(1),
                    dtype=attn_prefix.dtype,
                    device=device,
                )
                attention_mask = torch.cat([attn_prefix, attn_gen], dim=1)
            else:
                inputs_embeds = prefix_embeds
                attention_mask = prefix_attn

            outputs = self.llm_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
            logits = outputs.logits[:, -1, :]
            if neg_inf_value is None:
                neg_inf_value = torch.finfo(logits.dtype).min
            logits = logits / temperature
            logits = apply_repetition_penalty(logits, generated_ids, repetition_penalty)

            # Avoid immediate termination until min_new_tokens is reached; BOS/EOS share the same id in this setup.
            if step_idx < min_new_tokens:
                logits[:, eos_token_id] = neg_inf_value

            if finished.any():
                logits[finished, :] = neg_inf_value
                logits[finished, eos_token_id] = 0

            if top_p is not None and 0.0 < top_p < 1.0:
                probs = torch.softmax(logits, dim=-1)
                sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                mask = cumulative_probs > top_p
                mask[..., 1:] = mask[..., :-1].clone()
                mask[..., 0] = 0
                sorted_probs = sorted_probs.masked_fill(mask, 0.0)
                sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

                next_indices = torch.multinomial(sorted_probs, num_samples=1)
                next_tokens = sorted_indices.gather(-1, next_indices).squeeze(-1)
            else:
                next_tokens = torch.argmax(logits, dim=-1)

            generated_ids = torch.cat(
                [generated_ids, next_tokens.unsqueeze(-1)], dim=1
            )
            finished |= next_tokens.eq(eos_token_id)

            if finished.all():
                break

        texts = self.llm_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        cleaned = []
        for t in texts:
            ans = t.strip()
            if not ans:
                ans = "N/A"
            cleaned.append(ans)
        return cleaned

    @classmethod
    def from_config(cls, cfg):
        encoder_model = cfg.get("encoder_model")
        encoder_model_ckpt = cfg.get("encoder_model_ckpt", "")
        load_pretrained = cfg.get("load_pretrained", True)
        freeze_vit = cfg.get("freeze_vit", True)
        llm_model = cfg.get("llm_model")
        max_txt_len = cfg.get("max_txt_len", 64)
        num_query_token = cfg.get("num_query_token", 32)
        has_lora = cfg.get("has_lora", True)
        first_sentence_only = cfg.get("first_sentence_only", False)

        return cls(
            encoder_model=encoder_model,
            encoder_model_ckpt=encoder_model_ckpt,
            load_pretrained=load_pretrained,
            freeze_vit=freeze_vit,
            llm_model=llm_model,
            max_txt_len=max_txt_len,
            num_query_token=num_query_token,
            has_lora=has_lora,
            first_sentence_only=first_sentence_only,
        )
