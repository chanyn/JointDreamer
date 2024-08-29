<p align="center">
  <h2 align="center"><strong>[ECCV24] JointDreamer:<br> Ensuring Geometry Consistency and Text Congruence <br> in Text-to-3D Generation via Joint Score Distillation</strong></h2>

<p align="center">
    <a href="https://jiangchenhan.github.io/">Chenhan Jiang</a><sup>1*†</sup>,
    <a>Yihan Zeng</a><sup>2*</sup>,
    <a href="https://hu-tianyang.github.io/">Tianyang Hu</a><sup>2</sup>,
    <a>Songcun Xu</a><sup>2</sup>,
    <a>Wei Zhang</a><sup>3</sup>,
    <a href="https://xuhangcn.github.io/">Hang Xu</a><sup>4</sup><br>
    <a href="https://sites.google.com/view/dyyeung">Dit-Yan Yeung</a><sup>1</sup>,
    <br>
    <sup>†</sup>Corresponding authors.
    <br>
    <sup>1</sup>The Hong Kong University of Science and Technology,
    <sup>2</sup>Huawei Noah’s Ark Lab,
</p>


<div align="center">

<a href='https://arxiv.org/abs/2407.12291'><img src='https://img.shields.io/badge/arXiv-2407.12291-b31b1b.svg'></a> &nbsp;&nbsp;&nbsp;&nbsp;
 <a href='https://jointdreamer.github.io/'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp;&nbsp;&nbsp;&nbsp;

</div>


<p align="center">
    <img src="docs/demo_video.gif" alt="Demo GIF" width="512px" />
</p>


## Release
- [8/28] 🔥🔥 We released the **JointDreamer**.
- [7/01] The paper is now accepted to ECCV 2024.

## Contents
- [Installation](#installation)
- [Usage](#usage)
- [TODO](#todo)
- [Acknowledgement](#acknowledgement)
- [BibTeX](#bibtex)

## Installation
The environment follows the official [threestudio](https://github.com/threestudio-project/threestudio), skip it if you already have installed the environment. 
Next, install MVDream library with the following command:
```
# evaluate on python 3.10, pytorch 2.0.1, cuda 11.7
pip install -r requirements.txt
pip install -e extern/MVDream 
```

## Usage

### Text-to-3D Generation
```
# PROMPT = "A DSLR photo of a fox working on a jigsaw puzzle, 8K, HD, photorealistic"
# 3D generation with MVDream as view-awared model capturing view coherency.
python launch.py --config configs/jointdreamer-5k.yaml --train --gpu [GPU id] system.prompt_processor.prompt=[PROMPT]

# Refine texture with CFG switching and Geo Fading strategy
python launch.py --config configs/jointdreamer-refine.yaml --train --gpu [GPU id] system.prompt_processor.prompt=[PROMPT] resume=[path of the above folder]/ckpts/last.ckpt

# DMTet refinement
python launch.py --gpu [GPU id] --train --config configs/jointdreamer-dmtet.yaml system.prompt_processor.prompt=[PROMPT] system.geometry_convert_from=[path of the above folder]/ckpts/last.ckpt 

```
#### Tips for better performance
- Negative prompts tend to provide a clearer texture, but they can sometimes negatively impact the shape. As a general rule, we set ```start_neg=3000``` in ```prompt_processor```. Setting it to ```5000``` can sometimes yield a better shape.
- The strength of a view-aware model can influence view coherence. A typical setting is ```extra_guid_wt=5``` in ```guidance```. A higher value will enhance view coherence, but it may also result in floating elements and unclear textures at times.

### Classifier Training
```
# Note: the code follows DINOv2

# Data processing, produce the negative pairs.
python scripts/generate_pair.py

# Train the classifier
python classifier_train.py --data_path [Your data root] --num_pairs 1000000
```

## TODO

The repo is still being under construction, thanks for your patience. 
- [ ] Release of JSD with image-to-image translation models.
- [ ] Release of JSD with classifier model.

## Acknowledgement

Our code is based on these wonderful repos:

* [threestudio](https://github.com/threestudio-project/threestudio)
* [MVDream](https://github.com/bytedance/MVDream)

## BibTeX
```
@inproceedings{jiang2024jointdreaner,
  author = {Jiang, Chenhan and Zeng, Yihan and Hu, Tianyang and Xu, Songcun and Zhang, Wei and Xu, Wei and Yeung, Dit-Yan},
  title = {JointDreamer: Ensuring Geometry Consistency and Text Congruence in Text-to-3D Generation via Joint Score Distillation},
  booktitle = {ECCV},
  year = {2024},
}
```
