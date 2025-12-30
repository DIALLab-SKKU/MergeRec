# MergeRec: Model Merging for Data-Isolated Cross-Domain Sequential Recommendation (KDD 2025)

This repository provides the official implementation of MergeRec.

Paper: [TBD]()

## Datasets

Unzip and place the compressed datasets in the `datasets/` directory.
```bash
for FILE in $(ls datasets/*.tar.gz); do
  tar -xzvf $FILE -C datasets/
done
```

## Reproduction

The following sections describe how to reproduce the main results of MergeRec.

### 1. Requirements

Run the following command to initialize the conda environment:

```bash
conda env create -n mergerec
conda activate mergerec
pip install -r requirements.txt
```

### 2. Training & Evaluation

#### 2.1. Fine-tuning

The models need to be fine-tuned on each domain before merging.
Refer to `.sh` files at `scripts/finetune/` for examples of fine-tuning commands.

<details>
<summary>Information on pre-trained Recformer checkpoints</summary>
Pre-trained checkpoints for Recformer models can be accessed at the following links:
- Recformer-base: https://drive.google.com/file/d/1HRSS666INKPap6DftNLYjdXTPexLfDLW/view?usp=sharing
- Recformer-large: https://drive.google.com/file/d/1V32VTBcB0tIfUz7IEsSGzBW-AyiFRQFc/view?usp=sharing
</details>

<details>
<summary>Information on pre-trained BLaIR checkpoints</summary>
Pre-trained checkpoints for BLaIR-base is available at HuggingFace, so no additional download is needed.
However, you may have to gain access permission first: [HuggingFace](https://huggingface.co/hyp1231/blair-roberta-base)
</details>


#### 2.2. Preparing Item Embeddings

Before merging, we need to extract item embeddings for each domain.
Refer to `.sh` files at `scripts/embeddings/` for examples of item embedding extraction commands.

#### 2.3. Merging

After fine-tuning and pre-calculating item embeddings, we can merge the models.
Refer to `.sh` files at `scripts/mergerec/` for examples of merging commands.

> [!NOTE]
> Examples of baseline execution files are in the `scripts/baselines/` folder.

> [!NOTE]
> Run `merge_train.py -h`, `finetune_train.py -h` for more details about the arguments.


## Citation

If you find this work useful in your research, please consider citing:

```
TBD
```
