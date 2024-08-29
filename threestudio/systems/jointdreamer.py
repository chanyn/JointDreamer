from dataclasses import dataclass, field

import torch

import threestudio
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.ops import binary_cross_entropy, dot
from threestudio.utils.typing import *

import torch.nn.functional as F

@threestudio.register("jointdreamer-system")
class JointDreamer(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        guidance_eval_step: int = 200
        stage: str = 'coarse'

    cfg: Config

    def configure(self):
        # create geometry, material, background, renderer
        super().configure()

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        render_out = self.renderer(**batch)
        return {
            **render_out,
        }

    def on_fit_start(self) -> None:
        super().on_fit_start()
        # only used in training
        self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
            self.cfg.prompt_processor
        )
        self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)

    def training_step(self, batch, batch_idx):
        out = self(batch)
        prompt_utils = self.prompt_processor(self.true_global_step)

        guidance_eval = (self.true_global_step % self.cfg.guidance_eval_step == 0) and self.true_global_step > 0

        guidance_out = self.guidance(
            out["comp_rgb"], prompt_utils, **batch, opacity=out["opacity"], rgb_as_latents=False,
            guidance_eval=guidance_eval, global_step=self.true_global_step, normal=out.get("comp_normal", None),
        )

        if guidance_eval:
            for name, value in guidance_out["eval"].items():
                if name.startswith("imgs_"):
                    guidance_out["eval"][name] = value.detach()[: guidance_out["eval"]["bs"]]
            self.guidance_evaluation_save(
                out["comp_rgb"].detach()[: guidance_out["eval"]["bs"]],
                guidance_out["eval"],
            )
            del guidance_out["eval"]

        loss = 0.0

        for name, value in guidance_out.items():
            self.log(f"train/{name}", value)
            if name.startswith("loss_"):
                loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])

        if self.C(self.cfg.loss.lambda_orient) > 0:
            if "normal" not in out:
                raise ValueError(
                    "Normal is required for orientation loss, no normal is found in the output."
                )
            loss_orient = (
                out["weights"].detach()
                * dot(out["normal"], out["t_dirs"]).clamp_min(0.0) ** 2
            ).sum() / (out["opacity"] > 0).sum()
            self.log("train/loss_orient", loss_orient)
            loss += loss_orient * self.C(self.cfg.loss.lambda_orient)

        loss_sparsity = (out["opacity"] ** 2 + 0.01).sqrt().mean()
        self.log("train/loss_sparsity", loss_sparsity)
        loss += loss_sparsity * self.C(self.cfg.loss.lambda_sparsity)

        opacity_clamped = out["opacity"].clamp(1.0e-3, 1.0 - 1.0e-3)
        loss_opaque = binary_cross_entropy(opacity_clamped, opacity_clamped)
        self.log("train/loss_opaque", loss_opaque)
        loss += loss_opaque * self.C(self.cfg.loss.lambda_opaque)

        if self.C(self.cfg.loss.lambda_z_variance) > 0:
            loss_z_variance = out["z_variance"][out["opacity"] > 0.5].mean()
            self.log("train/loss_z_variance", loss_z_variance)
            loss += loss_z_variance * self.C(self.cfg.loss.lambda_z_variance)

        if self.C(self.cfg.loss.lambda_normal_consistency) > 0:
            loss_normal_consistency = out["mesh"].normal_consistency()
            self.log("train/loss_normal_consistency", loss_normal_consistency)
            loss += loss_normal_consistency * self.C(
                self.cfg.loss.lambda_normal_consistency
            )

        if self.C(self.cfg.loss.lambda_laplacian_smoothness) > 0:
            loss_laplacian_smoothness = out["mesh"].laplacian()
            self.log("train/loss_laplacian_smoothness", loss_laplacian_smoothness)
            loss += loss_laplacian_smoothness * self.C(
                self.cfg.loss.lambda_laplacian_smoothness
            )

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        self.save_image_grid(
            f"it{self.true_global_step}-{batch['index'][0]}.png",
            [
                {
                    "type": "rgb",
                    "img": out["comp_rgb"][0],
                    "kwargs": {"data_format": "HWC"},
                },
            ]
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            )
            + [
                {
                    "type": "grayscale",
                    "img": out["opacity"][0, :, :, 0],
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                },
            ],
            name="validation_step",
            step=self.true_global_step,
        )

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        out = self(batch)
        self.save_image_grid(
            f"it{self.true_global_step}-test/{batch['index'][0]}.png",
            [
                {
                    "type": "rgb",
                    "img": out["comp_rgb"][0],
                    "kwargs": {"data_format": "HWC"},
                },
            ]
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            ),
            name="test_step",
            step=self.true_global_step,
        )

    def on_test_epoch_end(self):
        self.save_img_sequence(
            f"it{self.true_global_step}-test",
            f"it{self.true_global_step}-test",
            "(\d+)\.png",
            save_format="mp4",
            fps=30,
            name="test",
            step=self.true_global_step,
        )

    def guidance_evaluation_save(self, comp_rgb, guidance_eval_out):
        B, size = comp_rgb.shape[:2]
        resize = lambda x: F.interpolate(
            x.permute(0, 3, 1, 2), (size, size), mode="bilinear", align_corners=False
        ).permute(0, 2, 3, 1)
        filename = f"it{self.true_global_step}-train.png"

        def merge12(x):
            return x.reshape(-1, *x.shape[2:])

        self.save_image_grid(
            filename,
            [
                {
                    "type": "rgb",
                    "img": merge12(comp_rgb),
                    "kwargs": {"data_format": "HWC"},
                },
            ]
            + (
                [
                    {
                        "type": "rgb",
                        "img": merge12(resize(guidance_eval_out["imgs_noisy"])),
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": merge12(resize(guidance_eval_out["imgs_1step"])),
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
            )
            +
            (
                [
                    {
                        "type": "rgb",
                        "img": merge12(resize(guidance_eval_out["imgs_1step_guidance"])),
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
                if "imgs_1step_guidance" in guidance_eval_out
                else []
            )
            +
            (
                [
                    {
                        "type": "rgb",
                        "img": merge12(resize(guidance_eval_out["imgs_1step_update"])),
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
                if "imgs_1step_update" in guidance_eval_out
                else []
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": merge12(resize(guidance_eval_out["imgs_1orig"])),
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
            )
            +
            (
                [
                    {
                        "type": "rgb",
                        "img": merge12(resize(guidance_eval_out["imgs_1orig_update"])),
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "imgs_1orig_update" in guidance_eval_out
                else []
            )
            +
            (
                [
                    {
                        "type": "rgb",
                        "img": merge12(resize(guidance_eval_out["imgs_1step_uncond"])),
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "imgs_1step_uncond" in guidance_eval_out
                else []
            ),
            name="train_step",
            step=self.true_global_step,
            texts=guidance_eval_out["texts"],
        )


@threestudio.register("jointdreamer-refine-system")
class JointDreamerRefine(JointDreamer):
    def training_step(self, batch, batch_idx):
        out = self(batch)
        prompt_utils = self.prompt_processor()

        guidance_eval = (self.true_global_step % self.cfg.guidance_eval_step == 0) and self.true_global_step > 0

        guidance_out = self.guidance(
            out["comp_rgb"], prompt_utils, **batch, opacity=out["opacity"], rgb_as_latents=False,
            guidance_eval=guidance_eval, global_step=self.true_global_step,
        )

        if guidance_eval:
            for name, value in guidance_out["eval"].items():
                if name.startswith("imgs_"):
                    guidance_out["eval"][name] = value.detach()[: guidance_out["eval"]["bs"]]
            self.guidance_evaluation_save(
                out["comp_rgb"].detach()[: guidance_out["eval"]["bs"]],
                guidance_out["eval"],
            )
            del guidance_out["eval"]

        loss = 0.0

        for name, value in guidance_out.items():
            self.log(f"train/{name}", value)
            if name.startswith("loss_"):
                loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])

        if self.cfg.stage == "coarse":
            if self.C(self.cfg.loss.lambda_orient) > 0:
                if "normal" not in out:
                    raise ValueError(
                        "Normal is required for orientation loss, no normal is found in the output."
                    )
                loss_orient = (
                                  out["weights"].detach()
                                  * dot(out["normal"], out["t_dirs"]).clamp_min(0.0) ** 2
                              ).sum() / (out["opacity"] > 0).sum()
                self.log("train/loss_orient", loss_orient)
                loss += loss_orient * self.C(self.cfg.loss.lambda_orient)

            loss_sparsity = (out["opacity"] ** 2 + 0.01).sqrt().mean()
            self.log("train/loss_sparsity", loss_sparsity)
            loss += loss_sparsity * self.C(self.cfg.loss.lambda_sparsity)

            opacity_clamped = out["opacity"].clamp(1.0e-3, 1.0 - 1.0e-3)
            loss_opaque = binary_cross_entropy(opacity_clamped, opacity_clamped)
            self.log("train/loss_opaque", loss_opaque)
            loss += loss_opaque * self.C(self.cfg.loss.lambda_opaque)

            if self.C(self.cfg.loss.lambda_z_variance) > 0:
                loss_z_variance = out["z_variance"][out["opacity"] > 0.5].mean()
                self.log("train/loss_z_variance", loss_z_variance)
                loss += loss_z_variance * self.C(self.cfg.loss.lambda_z_variance)

        elif self.cfg.stage == "texture":
            loss_normal_consistency = out["mesh"].normal_consistency()
            self.log("train/loss_normal_consistency", loss_normal_consistency)
            loss += loss_normal_consistency * self.C(
                self.cfg.loss.lambda_normal_consistency
            )
            if self.C(self.cfg.loss.lambda_laplacian_smoothness) > 0:
                loss_laplacian_smoothness = out["mesh"].laplacian()
                self.log("train/loss_laplacian_smoothness", loss_laplacian_smoothness)
                loss += loss_laplacian_smoothness * self.C(
                    self.cfg.loss.lambda_laplacian_smoothness
                )
        else:
            raise ValueError(f"Unknown stage {self.cfg.stage}")

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        return {"loss": loss}

    def test_step(self, batch, batch_idx):
        out = self(batch)
        self.save_rgba_image(
            f"it{self.true_global_step}-test/{batch['index'][0]}.png",
            out["comp_rgb"][0],
            out["opacity"][0, ..., 0],
            data_format="HWC",
            data_range=(0, 1),
        )
        self.save_rgba_image(
            f"it{self.true_global_step}-normal-test/{batch['index'][0]}.png",
            out["comp_normal"][0],
            out["opacity"][0, ..., 0],
            data_format="HWC",
            data_range=(0, 1),
        )

    def on_test_epoch_end(self):
        self.save_img_sequence(
            "demo",
            f"it{self.true_global_step}-test",
            "(\d+)\.png",
            save_format="mp4",
            fps=30,
            name="test",
            step=self.true_global_step,
        )
        self.save_img_sequence(
            "demo-normal",
            f"it{self.true_global_step}-normal-test",
            "(\d+)\.png",
            save_format="mp4",
            fps=30,
            name="test",
            step=self.true_global_step,
        )
