# WSD Model Based on Single-Twin Hybrid Tower Structure

## Introduction

This model solve the problem that there is no interaction between context and gloss in the traditional twin-tower model for the **Word Sense Disambiguation  (WSD)** task, proposing a **single-twin-tower hybrid model**. Inspired by [ALBEF](https://arxiv.org/abs/2107.07651) , I concatenates the context and gloss together and input them into the Transformer encoders, so that the newly obtained context and gloss representations have learnt the **interaction** between each other. And then I use the original loss function to update the gradient; at the same time, I use two loss settings, one using the output of single tower to calculate the loss, and the other using the outputs of the single tower and twin tower together to calculate the Loss. Finally, the f1 score of the hybrid model using both outputs of single and dual towers reaches **78.2** on the [**Sensevel2**](http://lcl.uniroma1.it/wsdeval/evaluation-data) dataset, which is 0.4% higher than SOTA--[wsd-biencoders](https://github.com/facebookresearch/wsd-biencoders).

The structure illustraion is below:![model_structure](https://github.com/llxblhyvia/Single-Twin-Tower-WSD/blob/main/model_structure.png).

## Dependencies

To run this code, you'll need the following libraries:

- Python 3
- Pytorch 1.2.0
- Pytorch Transformers 1.1.0
- Numpy 1.17.2
- NLTK 3.4.5
- tqdm
- and [WSD Evaluation Framework](http://lcl.uniroma1.it/wsdeval/home) which is standard evaluation framework for WSD task.

## Architecture

- WSD_Evaluation_Framework        # standard WSD_Evaluation_Framework downloaded for evaluation locally
- wsd-biencoders-main            # wsd package dir
  - yuanmodel.py   # original wsd-biencoders model for comparison
  - singleloss.py   # just use single tower output for training and evaluating
  - doublelosssingleeval.py  # use both the single tower and biencoder loss for training and just use single tower output for evaluating
  - doublelossdoubleeval.py        # use both the single tower and biencoder loss for training and evaluating
  -  wsd_models
      - models.py       # context, gloss, biencoder encoders
      - utils.py        # tokenizer, data processing, data loader, etc.
## How to Run

Firstly **compile** [Scorer.java]() on your server(`sudo` authorization needed) with:

```bash
javac ./WSD_Evaluation_Framework/Evaluation_Datasets/Scorer.java
```

**Train** model with:

```bash
python xxx.py --data-path $path_to_wsd_data --ckpt $path_to_checkpoint
```

with `$path_to_wsd_data` the directory  of `WSD_Evaluation_Framework`.

**Evaluate** model with:

```bash
python xxx.py --data-path $path_to_wsd_data --ckpt $path_to_model_checkpoint --eval --split $wsd_eval_set
```
with `$wsd_eval_set` the name of [evaluation datasets].

`xxx.py` is the training python file you choose this time.

### Notice: Running Time
Because of the use of Transformer blocks, the model training process will take several days according to your GPU for 20 epochs, so please use tools like tmux to keep the connection constantly.







