# Documen-Level Relation Extraction with NCRL
This Code is largely adapted from [ATLOP](https://github.com/wzhouad/ATLOP).

## Requirements
* Python (tested on 3.7.4)
* CUDA (tested on 11.2)
* [PyTorch](http://pytorch.org/) (tested on 1.7.0)
* [Transformers](https://github.com/huggingface/transformers) (tested on 3.4.0)
* numpy (tested on 1.18.5)
* [apex](https://github.com/NVIDIA/apex) (tested on 0.1)
* [opt-einsum](https://github.com/dgasmith/opt_einsum) (tested on 3.2.1)
* wandb
* ujson
* tqdm

## Dataset
The [DocRED](https://www.aclweb.org/anthology/P19-1074/) dataset can be downloaded following the instructions at [link](https://github.com/thunlp/DocRED/tree/master/data). The expected structure of files is:
```
DocRED
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
Train the BERT model on DocRED with the following command:

```bash
>> sh scripts/run_bert.sh  # for BERT
>> sh scripts/run_roberta.sh  # for RoBERTa
>> sh scripts/run_deberta.sh  # for DeBERTa
```

The training loss and evaluation results on the dev set are synced to the wandb dashboard.

The program will generate a test file `result.json` in the official evaluation format. You can compress and submit it to Colab for the official test score.

