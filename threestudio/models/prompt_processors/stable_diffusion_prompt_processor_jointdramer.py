from dataclasses import dataclass
import torch
import threestudio
from threestudio.utils.typing import *
from threestudio.models.prompt_processors.stable_diffusion_prompt_processor import StableDiffusionPromptProcessor
from threestudio.models.prompt_processors.base import PromptProcessorOutput
from threestudio.utils.misc import barrier
import os

def hash_prompt(model: str, prompt: str) -> str:
    import hashlib

    identifier = f"{model}-{prompt}"
    return hashlib.md5(identifier.encode()).hexdigest()

@threestudio.register("stable-diffusion-prompt-processor-jointdreamer")
class StableDiffusionPromptProcessorJD(StableDiffusionPromptProcessor):
    @dataclass
    class Config(StableDiffusionPromptProcessor.Config):
        start_neg: int = 0

    cfg: Config

    def configure(self) -> None:
        self._cache_dir = ".threestudio_cache/text_embeddings"  # FIXME: hard-coded path
        self.empty_prompt = ""
        cache_path = os.path.join(
            self._cache_dir,
            f"{hash_prompt(self.cfg.pretrained_model_name_or_path, self.empty_prompt)}.pt",
        )
        os.makedirs(self._cache_dir, exist_ok=True)
        if not os.path.exists(cache_path):
            self.spawn_func(
                self.cfg.pretrained_model_name_or_path,
                [self.empty_prompt],
                self._cache_dir,
            )
        super().configure()


    def load_text_embeddings(self):
        # synchronize, to ensure the text embeddings have been computed and saved to cache
        barrier()
        self.text_embeddings = self.load_from_cache(self.prompt)[None, ...] # bs,77,1024
        self.uncond_text_embeddings = self.load_from_cache(self.negative_prompt)[
            None, ...
        ]
        self.text_embeddings_vd = torch.stack(
            [self.load_from_cache(prompt) for prompt in self.prompts_vd], dim=0
        ) # 4,77,1024
        self.uncond_text_embeddings_vd = torch.stack(
            [self.load_from_cache(prompt) for prompt in self.negative_prompts_vd], dim=0
        )

        # record empty text embeddings
        self.empty_prompts_vd = [self.empty_prompt for _ in self.directions]

        self.empty_text_embeddings = self.load_from_cache(self.empty_prompt)[
            None, ...
        ]
        self.empty_text_embeddings_vd = torch.stack(
            [self.load_from_cache(prompt) for prompt in self.empty_prompts_vd], dim=0
        )
        threestudio.debug(f"Loaded text embeddings.")

    def __call__(self, global_step) -> PromptProcessorOutput:
        return PromptProcessorOutput(
            text_embeddings=self.text_embeddings,
            uncond_text_embeddings=self.uncond_text_embeddings if self.cfg.start_neg <= global_step else self.empty_text_embeddings,
            text_embeddings_vd=self.text_embeddings_vd,
            uncond_text_embeddings_vd=self.uncond_text_embeddings_vd if self.cfg.start_neg <= global_step else self.empty_text_embeddings_vd,
            directions=self.directions,
            direction2idx=self.direction2idx,
            use_perp_neg=self.cfg.use_perp_neg,
            perp_neg_f_sb=self.cfg.perp_neg_f_sb,
            perp_neg_f_fsb=self.cfg.perp_neg_f_fsb,
            perp_neg_f_fs=self.cfg.perp_neg_f_fs,
            perp_neg_f_sf=self.cfg.perp_neg_f_sf,
        )
