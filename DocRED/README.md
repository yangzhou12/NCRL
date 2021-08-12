# ATLOP
Code for AAAI 2021 paper [Document-Level Relation Extraction with Adaptive Thresholding and Localized Context Pooling](https://arxiv.org/abs/2010.11304).

If you make use of this code in your work, please kindly cite the following paper:

```bibtex
@inproceedings{zhou2021atlop,
	title={Document-Level Relation Extraction with Adaptive Thresholding and Localized Context Pooling},
	author={Zhou, Wenxuan and Huang, Kevin and Ma, Tengyu and Huang, Jing},
	booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
	year={2021}
}
```
## Requirements
* Python (tested on 3.7.4)
* CUDA (tested on 10.2)
* [PyTorch](http://pytorch.org/) (tested on 1.7.0)
* [Transformers](https://github.com/huggingface/transformers) (tested on 3.4.0)
* numpy (tested on 1.19.4)
* [apex](https://github.com/NVIDIA/apex) (tested on 0.1)
* [opt-einsum](https://github.com/dgasmith/opt_einsum) (tested on 3.3.0)
* wandb
* ujson
* tqdm

## Dataset
The [DocRED](https://www.aclweb.org/anthology/P19-1074/) dataset can be downloaded following the instructions at [link](https://github.com/thunlp/DocRED/tree/master/data). The expected structure of files is:
```
ATLOP
 |-- dataset
 |    |-- docred
 |    |    |-- train_annotated.json        
 |    |    |-- train_distant.json
 |    |    |-- dev.json
 |    |    |-- test.json
 |-- meta
 |    |-- rel2id.json
```

## Training and Evaluation
### DocRED
Train the BERT model on DocRED with the following command:

```bash
>> sh scripts/run_bert.sh  # for BERT
>> sh scripts/run_roberta.sh  # for RoBERTa
```

The training loss and evaluation results on the dev set are synced to the wandb dashboard.

The program will generate a test file `result.json` in the official evaluation format. You can compress and submit it to Colab for the official test score.
