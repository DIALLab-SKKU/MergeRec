# MergeRec: Model Merging for Data-Isolated Cross-Domain Sequential Recommendation (KDD 2026)

This repository provides the official implementation of MergeRec.

> [!NOTE]
> This repository is an archived snapshot provided for the paper artifact.
> 
> For the latest updates, please visit the [active repository](https://github.com/DIALLab-SKKU/MergeRec)

<details>
<summary>Abstract</summary>

Modern recommender systems trained on domain-specific data often struggle to generalize across multiple domains. Cross-domain sequential recommendation has emerged as a promising research direction to address this challenge; however, existing approaches face fundamental limitations, such as reliance on overlapping users or items across domains, or unrealistic assumptions that ignore privacy constraints.<br>
In this work, we propose a new framework, <b><i>MergeRec</i></b>, based on <i>model merging</i> under a new and realistic problem setting termed <i>data-isolated cross-domain sequential recommendation</i>, where raw user interaction data cannot be shared across domains.<br>
MergeRec consists of three key components: (1) <i>merging initialization</i>, (2) <i>pseudo-user data construction</i>, and (3) <i>collaborative merging optimization</i>. First, we initialize a merged model using training-free merging techniques. Next, we construct pseudo-user data by treating each item as a virtual sequence in each domain, enabling the synthesis of meaningful training samples without relying on real user interactions. Finally, we optimize domain-specific merging weights through a joint objective that combines a <i>recommendation loss</i> and a <i>distillation loss</i>.
</details>

Paper: [TBD]()

## Datasets

We provide preprocessed datasets for 8 domains in this repository.
Unzip the compressed datasets in the `datasets/` directory as follows:
```bash
for FILE in $(ls datasets/*.tar.gz); do
  tar -xzvf $FILE -C datasets/
done
```
The datasets used in this repository are derived from the Amazon Review Data released by the UCSD McAuley Lab, as listed below.
- [2014 version](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html): Beauty, Sports, and Toys
- [2018 version](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/): Arts, Instruments, Office, Pantry, and Scientific

The original datasets are publicly available from [the authors’ website](https://cseweb.ucsd.edu/~jmcauley/datasets.html#amazon_reviews).

## Experiment Reproduction

The following sections describe how to reproduce the experimental results.

### 1. Requirements

Run the following commands to initialize the `conda` environment:

```bash
conda create -n mergerec python=3.12 -y
conda activate mergerec
pip install -r requirements.txt
```

### 2. Reproduction Steps

> [!WARNING]
> If more than one GPU is installed on the system, use the environment variable `CUDA_VISIBLE_DEVICES` to specify only **one GPU**.
> The code supports single-GPU execution only.

> [!NOTE]
> To disable experiment tracking with `wandb`, set the environment variable `WANDB_MODE=disabled`

#### 2.1. Fine-tuning

The models are fine-tuned on each domain before merging.

For details, refer to the `.sh` files in `scripts/1_finetune/` for examples of fine-tuning commands.

<details>
<summary>Pre-trained Recformer checkpoints</summary>
Pre-trained Recformer checkpoints are available at the following links:
- Recformer-base: https://drive.google.com/file/d/1HRSS666INKPap6DftNLYjdXTPexLfDLW/view?usp=sharing
- Recformer-large: https://drive.google.com/file/d/1V32VTBcB0tIfUz7IEsSGzBW-AyiFRQFc/view?usp=sharing
</details>

<details>
<summary>Pre-trained BLaIR checkpoints</summary>
Pre-trained checkpoints for BLaIR-base are available on HuggingFace, and no additional download is required.
Access permission may be required: [HuggingFace](https://huggingface.co/hyp1231/blair-roberta-base)
</details>

##### Information on model identifiers

The fine-tuned model checkpoints will be saved in `MergeRecFineTune/{random_id}/checkpoints/`,
where `{random_id}` is an 8-character random string.
The last 4 characters of this ID are displayed in the progress bar during fine-tuning.


#### 2.2. Postprocessing model checkpoints

Before merging, each `pytorch-lightning` checkpoint is postprocessed as follows:
1. The `state_dict` is extracted.
2. The item embeddings are extracted.

Example:
```bash
python scripts/2_ft_postprocess/extract.py ./MergeRecFineTune/abcdefgh/epoch_00.ckpt ./checkpoints/blair_base/Arts/
```

#### 2.3. Merging

The models are now ready to be merged.

Refer to the `.sh` files in `scripts/3_mergerec/` for examples of merging commands.

> [!NOTE]
> Examples of baseline scripts are in the `scripts/baselines/` folder.

> [!NOTE]
> Run `python merge_train.py -h` or `python finetune_train.py -h` for a detailed description of the available arguments.
 
### Reproducibility Notes
- All merging experiments are conducted with 5 fixed random seeds.
- Results may vary slightly depending on GPU architecture and CUDA version.

## Citation

If you find this work useful, please consider citing our paper:

```bibtex
@inproceedings{kim2026mergerec,
  title     = {MergeRec: Model Merging for Data-Isolated Cross-Domain Sequential Recommendation},
  author    = {Kim, Hyunsoo and Moon, Jaewan and Park, Seongmin and Lee, Jongwuk},
  booktitle = {Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  year      = {2026},
  series    = {KDD '26},
  address   = {Jeju Island, Republic of Korea},
  month     = aug,
  publisher = {ACM},
  doi       = {10.1145/3770854.3780264},
  isbn      = {979-8-4007-2258-5}
}
```

## Contact
For questions or issues, please contact the authors via email.

## License
This project is licensed under the MIT License.
See the `LICENSE` file for details.
