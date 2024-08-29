from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


import threestudio
from threestudio.models.prompt_processors.base import PromptProcessorOutput
from threestudio.utils.typing import *

# add
from threestudio.models.guidance.stable_diffusion_guidance import StableDiffusionGuidance
from omegaconf import OmegaConf
from mvdream.model_zoo import build_model
from mvdream.camera_utils import normalize_camera


class MVDreamGuidance(nn.Module):
    def __init__(self, cfg, device):
        super().__init__()
        self.device = device
        self.weights_dtype = torch.float32
        self.cfg = cfg

        self.model = build_model(cfg.model_name, ckpt_path=cfg.ckpt)
        self.model.device = device
        self.model.to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    def encode_images(
        self, imgs: Float[Tensor, "B 3 256 256"]
    ) -> Float[Tensor, "B 4 32 32"]:
        imgs = imgs * 2.0 - 1.0
        latents = self.model.get_first_stage_encoding(self.model.encode_first_stage(imgs))
        return latents  # [B, 4, 32, 32] Latent space image

    def get_camera_cond(self,
                        camera: Float[Tensor, "B 4 4"],
                        ):
        # Note: the input of threestudio is already blender coordinate system
        # camera = convert_opengl_to_blender(camera)
        if self.cfg.camera_condition_type == "rotation":  # normalized camera
            camera = normalize_camera(camera)
            camera = camera.flatten(start_dim=1)
        else:
            raise NotImplementedError(f"Unknown camera_condition_type={self.cfg.camera_condition_type}")
        return camera

    def forward(self, latents_func, camera, text_embeddings, t):
        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents_func)
            latents_noisy = self.model.q_sample(latents_func, t, noise)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            # save input tensors for UNet
            camera = self.get_camera_cond(camera)
            camera = camera.repeat(2, 1).to(text_embeddings)
            context = {"context": text_embeddings, "camera": camera, "num_frames": self.cfg.n_view}

            noise_pred = self.model.apply_model(latent_model_input, torch.cat([t] * 2, dim=0), context)

        # perform guidance
        noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)  # Note: flipped compared to stable-dreamfusion
        noise_pred = noise_pred_uncond + self.cfg.guidance_scale * (noise_pred_text - noise_pred_uncond)

        latents_recon = self.model.predict_start_from_noise(latents_noisy, t, noise_pred)
        # clip or rescale x0
        if self.cfg.recon_std_rescale > 0:
            latents_recon_nocfg = self.model.predict_start_from_noise(latents_noisy, t, noise_pred_text)
            latents_recon_nocfg_reshape = latents_recon_nocfg.view(-1, self.cfg.n_view, *latents_recon_nocfg.shape[1:])
            latents_recon_reshape = latents_recon.view(-1, self.cfg.n_view, *latents_recon.shape[1:])
            factor = (latents_recon_nocfg_reshape.std([1, 2, 3, 4], keepdim=True) + 1e-8) / (
                    latents_recon_reshape.std([1, 2, 3, 4], keepdim=True) + 1e-8)

            latents_recon_adjust = latents_recon.clone() * factor.squeeze(1).repeat_interleave(self.cfg.n_view, dim=0)
            latents_recon = self.cfg.recon_std_rescale * latents_recon_adjust + (
                    1 - self.cfg.recon_std_rescale) * latents_recon

        return latents_recon

    def predict(self, latents_noisy, camera, text_embeddings, t):
        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            # save input tensors for UNet
            camera = self.get_camera_cond(camera)
            camera = camera.repeat(2, 1).to(text_embeddings)
            context = {"context": text_embeddings, "camera": camera, "num_frames": self.cfg.n_view}

            noise_pred = self.model.apply_model(latent_model_input, torch.cat([t] * 2, dim=0), context)

        # perform guidance
        noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)  # Note: flipped compared to stable-dreamfusion
        noise_pred = noise_pred_uncond + self.cfg.guidance_scale * (noise_pred_text - noise_pred_uncond)

        latents_recon = self.model.predict_start_from_noise(latents_noisy, t, noise_pred)
        # clip or rescale x0
        if self.cfg.recon_std_rescale > 0:
            latents_recon_nocfg = self.model.predict_start_from_noise(latents_noisy, t, noise_pred_text)
            latents_recon_nocfg_reshape = latents_recon_nocfg.view(-1, self.cfg.n_view, *latents_recon_nocfg.shape[1:])
            latents_recon_reshape = latents_recon.view(-1, self.cfg.n_view, *latents_recon.shape[1:])
            factor = (latents_recon_nocfg_reshape.std([1, 2, 3, 4], keepdim=True) + 1e-8) / (
                    latents_recon_reshape.std([1, 2, 3, 4], keepdim=True) + 1e-8)

            latents_recon_adjust = latents_recon.clone() * factor.squeeze(1).repeat_interleave(self.cfg.n_view, dim=0)
            latents_recon = self.cfg.recon_std_rescale * latents_recon_adjust + (
                    1 - self.cfg.recon_std_rescale) * latents_recon

        return latents_recon

    def loss(self, latents, latents_recon, mask=None):
        if latents_recon.shape[2] != latents.shape[2]:
            latents_recon = F.interpolate(latents_recon, latents.shape[2:], mode='bilinear', align_corners=False)
        if mask is not None:
            mask = F.interpolate(mask, latents_recon.shape[2:], mode='bilinear', align_corners=False)
            latents_recon = latents_recon * mask
        l = 0.5 * F.mse_loss(latents, latents_recon.detach(), reduction="sum") / latents.shape[0]
        return l


@threestudio.register("interview-diffusion-guidance-mvdream")
class InterViewDiffusionGuidance(StableDiffusionGuidance):
    @dataclass
    class Config(StableDiffusionGuidance.Config):
        guide: str = ''
        extra_guid_wt: float = 1.0
        num_steps: int = 1
        guidance_type: str = "sds"
        weighting_strategy: str = "dreamfusion"
        start_iter: int = 0
        cfg_rescale: float = 0.0

    cfg: Config

    def configure(self) -> None:
        super().configure()
        # self.alphas: Float[Tensor, "..."] = self.scheduler.alphas_cumprod
        self.sigmas: Float[Tensor, "T"] = (1 - self.alphas) ** 0.5

        if self.cfg.guide:
            guide_model_config = OmegaConf.load(self.cfg.guide)
            self.guidance_model = MVDreamGuidance(guide_model_config, self.device)


    def get_noise_pred(
        self,
        latents_noisy,
        t,
        text_embeddings,
    ):
        with torch.no_grad():
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
            noise_pred = self.forward_unet(
                latent_model_input,
                torch.cat([t] * 2, dim=0),
                encoder_hidden_states=text_embeddings,
            )

        # perform guidance (high scale from paper!)
        noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
        noise_pred = noise_pred_text + self.cfg.guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )

        if self.cfg.cfg_rescale > 0:
            std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
            std_pred = noise_pred.std(dim=list(range(1, noise_pred.ndim)), keepdim=True)
            noise_pred_rescaled = noise_pred * (std_text / std_pred)
            # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
            noise_pred = (
                self.cfg.cfg_rescale * noise_pred_rescaled + (1 - self.cfg.cfg_rescale) * noise_pred
            )

        return noise_pred, noise_pred_uncond


    def decode_x0_1step(self, t, latents_noisy, eps_pred):
        alpha_prod_t = self.alphas[t]
        beta_prod_t = 1 - alpha_prod_t
        latents_1step = []
        for a_t, b_t, latent_t, eps_t in zip(alpha_prod_t, beta_prod_t, latents_noisy, eps_pred):
            latents_1step_t = (latent_t - b_t ** (0.5) * eps_t) / a_t ** (0.5)
            latents_1step.append(latents_1step_t.unsqueeze(0))

        return torch.cat(latents_1step, dim=0)


    def update_by_guidance(self, eps_pred, eps_pred_uncond, t, latents_noisy, rgb_func, text_embeddings,
                           cam_transform, opacity):
        opacity = F.interpolate(opacity, latents_noisy.shape[2:], mode='bilinear', align_corners=False)
        lantent_denoise = self.decode_x0_1step(t, latents_noisy * opacity, eps_pred_uncond)

        # calculate guidance
        latents_guide = self.guidance_model.encode_images(rgb_func)
        lantent_denoise_guide = self.guidance_model(latents_guide, cam_transform, text_embeddings, t)

        selected = -1 * self.guidance_model.loss(lantent_denoise, lantent_denoise_guide)
        grad = torch.autograd.grad(selected.sum(), latents_noisy)[0]
        grad = grad * self.cfg.extra_guid_wt

        eps_pred = eps_pred - self.sigmas[t].view(-1, 1, 1, 1) * grad.detach()

        guidance_eval_out = {"latent_1step_uncond": lantent_denoise,
                             "latent_1step_guide": lantent_denoise_guide,
                             "noise_pred_updated": eps_pred}

        return eps_pred, guidance_eval_out

    def __call__(
        self,
        rgb: Float[Tensor, "B H W C"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        c2w: Float[Tensor, "B 4 4"],
        opacity: Float[Tensor, "B H W 1"],
        rgb_as_latents=False,
        guidance_eval=False,
        **kwargs,
    ):
        global_step = kwargs.get("global_step", -1)
        batch_size = rgb.shape[0]
        camera = c2w

        rgb_BCHW = rgb.permute(0, 3, 1, 2)
        opacity = opacity.permute(0, 3, 1, 2)

        latents: Float[Tensor, "B 4 64 64"]
        if rgb_as_latents:
            latents = F.interpolate(
                rgb_BCHW, (64, 64), mode="bilinear", align_corners=False
            )
        else:
            rgb_BCHW_512 = F.interpolate(
                rgb_BCHW, (512, 512), mode="bilinear", align_corners=False
            )
            # encode image into latents with vae
            latents = self.encode_images(rgb_BCHW_512)

            # mvdream latents
            rgb_func = F.interpolate(rgb_BCHW, (256, 256), mode='bilinear', align_corners=False)


        t = torch.randint(
            self.min_step,
            self.max_step + 1,
            [1],
            dtype=torch.long,
            device=self.device,
        ).repeat(batch_size)

        # compute text emb
        if prompt_utils.use_perp_neg:
            (text_embeddings, neg_guidance_weights,) = prompt_utils.get_text_embeddings_perp_neg(
                elevation, azimuth, camera_distances, self.cfg.view_dependent_prompting)
        else:
            text_embeddings = prompt_utils.get_text_embeddings(
                elevation, azimuth, camera_distances, self.cfg.view_dependent_prompting
            )
            neg_guidance_weights = None

        # sample noise
        noise = torch.randn_like(latents)
        latents_noisy = self.scheduler.add_noise(latents, noise, t)
        eps_pred, eps_pred_uncond = self.get_noise_pred(
            latents_noisy, t, text_embeddings)

        # compute SDS gradient
        if self.cfg.guidance_type == "sds":
            eps_phi = noise
        else:
            raise ValueError("Not implemented")

        if self.cfg.weighting_strategy == "dreamfusion":
            w = (1.0 - self.alphas[t]).view(-1, 1, 1, 1)
        elif self.cfg.weighting_strategy == "uniform":
            w = 1.0
        elif self.cfg.weighting_strategy == "fantasia3d":
            w = (self.alphas[t] ** 0.5 * (1 - self.alphas[t])).view(-1, 1, 1, 1)
        else:
            raise ValueError(
                f"Unknown weighting strategy: {self.cfg.weighting_strategy}"
            )

        # compute x0_ref_1step: latent_ref -> get_eps -> denoise_1step -> decode
        if self.cfg.guide and global_step > self.cfg.start_iter:
            latents_noisy_in = latents_noisy.detach().requires_grad_(True)
            eps_pred_new, guidance_eval_utils_guid = self.update_by_guidance(eps_pred, eps_pred_uncond.detach(), t,
                                                                             latents_noisy_in, rgb_func,
                                                                             text_embeddings, camera, opacity)

            grad = w * (eps_pred_new - eps_phi)
        else:
            grad = w * (eps_pred - eps_phi)

        grad = torch.nan_to_num(grad)
        # clip grad for stable training?
        if self.grad_clip_val is not None:
            grad = grad.clamp(-self.grad_clip_val, self.grad_clip_val)

        # reparameterization trick:
        # d(loss)/d(latents) = latents - target = latents - (latents - grad) = grad
        target = (latents - grad).detach()
        loss_sds = 0.5 * F.mse_loss(latents, target, reduction="sum") / batch_size


        guidance_out = {
            "loss_sds": loss_sds,
            "grad_norm": grad.norm(),
            "min_step": self.min_step,
            "max_step": self.max_step,
        }

        if guidance_eval:
            guidance_eval_utils = {
                "neg_guidance_weights": neg_guidance_weights,
                "text_embeddings": text_embeddings,
                "t_orig": t,
                "latents_noisy": latents_noisy,
                "noise_pred": eps_pred,
            }
            if self.cfg.guide and global_step > self.cfg.start_iter:
                guidance_eval_utils.update(guidance_eval_utils_guid)

            guidance_eval_out = self.guidance_eval(**guidance_eval_utils)

            texts = []
            for n, e, a, c in zip(
                guidance_eval_out["noise_levels"], elevation, azimuth, camera_distances
            ):
                texts.append(
                    f"n{n:.02f}\ne{e.item():.01f}\na{a.item():.01f}\nc{c.item():.02f}"
                )
            guidance_eval_out.update({"texts": texts})
            guidance_out.update({"eval": guidance_eval_out})

        return guidance_out

    @torch.cuda.amp.autocast(enabled=False)
    @torch.no_grad()
    def guidance_eval(
        self,
        t_orig,
        text_embeddings,
        latents_noisy,
        noise_pred,
        use_perp_neg=False,
        neg_guidance_weights=None,
        noise_pred_updated=None,
        latent_1step_guide=None,
        latent_1step_uncond=None,
    ):
        # use only 50 timesteps, and find nearest of those to t
        self.scheduler.set_timesteps(50)
        self.scheduler.timesteps_gpu = self.scheduler.timesteps.to(self.device)
        bs = (
            min(self.cfg.max_items_eval, latents_noisy.shape[0])
            if self.cfg.max_items_eval > 0
            else latents_noisy.shape[0]
        )  # batch size
        large_enough_idxs = self.scheduler.timesteps_gpu.expand([bs, -1]) > t_orig[
            :bs
        ].unsqueeze(
            -1
        )  # sized [bs,50] > [bs,1]
        idxs = torch.min(large_enough_idxs, dim=1)[1]
        t = self.scheduler.timesteps_gpu[idxs]

        fracs = list((t / self.scheduler.config.num_train_timesteps).cpu().numpy())
        imgs_noisy = self.decode_latents(latents_noisy[:bs]).permute(0, 2, 3, 1)

        # get prev latent
        latents_1step = []
        pred_1orig = []
        latents_1step_update = []
        pred_1orig_update = []
        for b in range(bs):
            step_output = self.scheduler.step(
                noise_pred[b : b + 1], t[b], latents_noisy[b : b + 1], eta=1
            )
            latents_1step.append(step_output["prev_sample"])
            pred_1orig.append(step_output["pred_original_sample"])

            if noise_pred_updated is not None:
                step_output_update = self.scheduler.step(
                    noise_pred_updated[b: b + 1], t[b], latents_noisy[b: b + 1], eta=1
                )
                latents_1step_update.append(step_output_update["prev_sample"])
                pred_1orig_update.append(step_output_update["pred_original_sample"])

        latents_1step = torch.cat(latents_1step)
        pred_1orig = torch.cat(pred_1orig)
        imgs_1step = self.decode_latents(latents_1step).permute(0, 2, 3, 1)
        imgs_1orig = self.decode_latents(pred_1orig).permute(0, 2, 3, 1)

        out = {
            "bs": bs,
            "noise_levels": fracs,
            "imgs_noisy": imgs_noisy,
            "imgs_1step": imgs_1step,
            "imgs_1orig": imgs_1orig,
        }
        #####################
        if noise_pred_updated is not None:
            latents_1step_update = torch.cat(latents_1step_update)
            pred_1orig_update = torch.cat(pred_1orig_update)
            imgs_1step_update = self.decode_latents(latents_1step_update).permute(0, 2, 3, 1)
            imgs_1orig_update = self.decode_latents(pred_1orig_update).permute(0, 2, 3, 1)
            out.update({"imgs_1step_update": imgs_1step_update,
                        "imgs_1orig_update": imgs_1orig_update})

        if latent_1step_guide is not None:
            imgs_1step_guide = self.decode_latents(latent_1step_guide).permute(0, 2, 3, 1)
            out.update({"imgs_1step_guidance": imgs_1step_guide})

        if latent_1step_uncond is not None:
            imgs_1step_uncond = self.decode_latents(latent_1step_uncond).permute(0, 2, 3, 1)
            out.update({"imgs_1step_uncond": imgs_1step_uncond})

        return out
