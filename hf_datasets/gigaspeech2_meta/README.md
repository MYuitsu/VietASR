---
language:
- th
- id
- vi
license:
- apache-2.0
multilinguality:
- multilingual
pretty_name: GigaSpeech 2
task_categories:
- automatic-speech-recognition
extra_gated_prompt: "SpeechColab does not own the copyright of the audio files. For\
  \ researchers and educators who wish to use the audio files for non-commercial research\
  \ and/or educational purposes, we can provide access through the Hub under certain\
  \ conditions and terms. \nTerms of Access:\nThe \"Researcher\" has requested permission\
  \ to use the GigaSpeech database (the \"Database\") at Tsinghua University. In exchange\
  \ for such permission, Researcher hereby agrees to the following terms and conditions:\n\
  1. Researcher shall use the Database only for non-commercial research and educational\
  \ purposes.\n2. The SpeechColab team and Tsinghua University make no representations\
  \ or warranties regarding the Database, including but not limited to warranties\
  \ of non-infringement or fitness for a particular purpose.\n3. Researcher accepts\
  \ full responsibility for his or her use of the Database and shall defend and indemnify\
  \ the SpeechColab team and Tsinghua University, including their employees, Trustees,\
  \ officers and agents, against any and all claims arising from Researcher's use\
  \ of the Database, including but not limited to Researcher's use of any copies of\
  \ copyrighted audio files that he or she may create from the Database.\n4. Researcher\
  \ may provide research associates and colleagues with access to the Database provided\
  \ that they first agree to be bound by these terms and conditions.\n5. The SpeechColab\
  \ team and Tsinghua University reserve the right to terminate Researcher's access\
  \ to the Database at any time.\n6. If Researcher is employed by a for-profit, commercial\
  \ entity, Researcher's employer shall also be bound by these terms and conditions,\
  \ and Researcher hereby represents that he or she is fully authorized to enter into\
  \ this agreement on behalf of such employer.\n\n!!!"
extra_gated_fields:
  Name: text
  Email: text
  Organization: text
  Address: text
  I hereby confirm that I have requested access via the Google Form provided above: checkbox
  I accept the terms of access: checkbox
dataset_info:
- config_name: th
  features:
  - name: __key__
    dtype: string
  - name: __url__
    dtype: string
  - name: wav
    dtype:
      audio:
        sampling_rate: 16000
  splits:
  - name: train
  - name: dev
  - name: test
- config_name: id
  features:
  - name: __key__
    dtype: string
  - name: __url__
    dtype: string
  - name: wav
    dtype:
      audio:
        sampling_rate: 16000
  splits:
  - name: train
  - name: dev
  - name: test
- config_name: vi
  features:
  - name: __key__
    dtype: string
  - name: __url__
    dtype: string
  - name: wav
    dtype:
      audio:
        sampling_rate: 16000
  splits:
  - name: train
  - name: dev
  - name: test
configs:
- config_name: th
  data_files:
  - split: train
    path: data/th/train/*.tar.gz
  - split: dev
    path: data/th/dev.tar.gz
  - split: test
    path: data/th/test.tar.gz
- config_name: id
  data_files:
  - split: train
    path: data/id/train/*.tar.gz
  - split: dev
    path: data/id/dev.tar.gz
  - split: test
    path: data/id/test.tar.gz
- config_name: vi
  data_files:
  - split: train
    path: data/vi/train/*.tar.gz
  - split: dev
    path: data/vi/dev.tar.gz
  - split: test
    path: data/vi/test.tar.gz
tags:
- croissant
size_categories:
- n>1T
---

# Dataset Card for GigaSpeech 2

## Table of Contents
- [Table of Contents](#table-of-contents)
- [Dataset Description](#dataset-description)
  - [Example Usage](#example-usage)
  - [Supported Tasks and Leaderboards](#supported-tasks-and-leaderboards)
  - [Languages](#languages)
- [Dataset Structure](#dataset-structure)
  - [Data Instances](#data-instances)
  - [Data Fields](#data-fields)
  - [Data Splits](#data-splits)
- [Dataset Creation](#dataset-creation)
  - [Source Data](#source-data)
  - [Annotations](#annotations)
  - [Licensing Information](#licensing-information)
  - [Citation Information](#citation-information)
- [Terms of Access](#terms-of-access)
  

## Dataset Description

GigaSpeech 2 is an evolving, large-scale, multi-domain, and multilingual ASR corpus focusing on low-resource languages. GigaSpeech 2 raw comprises about 30,000 hours of automatically transcribed speech, across Thai, Indonesian, and Vietnamese. GigaSpeech 2 refine consists of 10,000 hours of Thai, 6,000 hours each for Indonesian and Vietnamese.

- **Repository:** https://github.com/SpeechColab/GigaSpeech2
- **Paper:** https://aclanthology.org/2025.acl-long.135.pdf
- **Leaderboard:** https://github.com/SpeechColab/GigaSpeech2#leaderboard
- **ModelScope:** https://modelscope.cn/datasets/AI-ModelScope/gigaspeech2
- **Point of Contact:** [gigaspeech@speechcolab.org](mailto:gigaspeech@speechcolab.org)

### Preparation Scripts

```bash
pip install lhotse
lhotse prepare gigaspeech2 [OPTIONS] CORPUS_DIR OUTPUT_DIR
```

### Supported Tasks and Leaderboards

- `automatic-speech-recognition`: The dataset can be used to train a model for Automatic Speech Recognition (ASR). The model is presented with an audio file and asked to transcribe the audio file to written text. Evaluation metrics includes Character Error Rate (CER) for Thai, and Word Error Rate (WER) for Indonesian and Vietnamese. The task has an active leaderboard which can be found at https://github.com/SpeechColab/GigaSpeech2#leaderboard and ranks models based on their WER.

### Languages
Gigaspeech 2 contains audio and transcription data in Thai, Indonesian, and Vietnamese.

## Dataset Structure

```
GigaSpeech 2
├── data
│   ├── id
│   │   ├── md5
│   │   ├── dev.tar.gz
│   │   ├── dev.tsv
│   │   ├── test.tar.gz
│   │   ├── test.tsv
│   │   ├── train
│   │   │   ├── 0.tar.gz
│   │   │   ├── 1.tar.gz
│   │   │   └── ...
│   │   ├── train_raw.tsv
│   │   └── train_refined.tsv
│   ├── th
│   │   ├── md5
│   │   ├── dev.tar.gz
│   │   ├── dev.tsv
│   │   ├── test.tar.gz
│   │   ├── test.tsv
│   │   ├── train
│   │   │   ├── 0.tar.gz
│   │   │   ├── 1.tar.gz
│   │   │   └── ...
│   │   ├── train_raw.tsv
│   │   └── train_refined.tsv
│   └── vi
│       ├── md5
│       ├── dev.tar.gz
│       ├── dev.tsv
│       ├── test.tar.gz
│       ├── test.tsv
│       ├── train
│       │   ├── 0.tar.gz
│       │   ├── 1.tar.gz
│       │   └── ...
│       ├── train_raw.tsv
│       └── train_refined.tsv
├── metadata.json
└── README.md
```

### Data Instances
```
Audio file (.wav):
    Channels: 1
    Sample Rate: 16000
    Sample Encoding: 16-bit Signed Integer PCM

Transcript file (.tsv):
    <segment_id>\t<text>\n
```

### Data Fields
- segment_id (string) - string id of the segment.
- text (string) - transcription of the segment.

### Data Splits
The dataset has three subsets for each language: train, dev, and test.
The train set has two configurations: raw and refined. train_raw contains all the data from train_refined.

#### Transcribed Training Subsets Size

|                      | Thai (hours) | Indonesian (hours) | Vietnamese (hours) |
|:--------------------:|:------------:|:------------------:|:------------------:|
| GigaSpeech 2 raw     |    12901.8   |      8112.9        |      7324.0        |
| GigaSpeech 2 refined |    10262.0   |      5714.0        |      6039.0        |

GigaSpeech 2 raw contains all the data from GigaSpeech 2 refined.

#### Transcribed Evaluation Subsets

|                      | Thai (hours) | Indonesian (hours) | Vietnamese (hours) |
|:--------------------:|:------------:|:------------------:|:------------------:|
| GigaSpeech 2 dev     |     10.0     |       10.0         |       10.2         |
| GigaSpeech 2 test    |     10.0     |       10.0         |       11.0         |

## Dataset Creation

### Source Data
* GigaSpeech 2 raw: 30,000 hours of automatically transcribed speech across Thai, Indonesian, and Vietnamese.
* GigaSpeech 2 refined: 10,000 hours of Thai, 6,000 hours each for Indonesian and Vietnamese.
* GigaSpeech 2 DEV & TEST: 10 hours for DEV and 10 hours for TEST per language, **transcribed by professional human annotators**, challenging and realistic.

### Annotations

#### Who are the annotators?
Development (DEV) and test (TEST) subsets are annotated by professional human annotators.

### Licensing Information

SpeechColab does not own the copyright of the audio files. For researchers and educators who wish to use the audio files for 
non-commercial research and/or educational purposes, we can provide access through our site under certain conditions and terms.

In general, when training a machine learning model on a given dataset, the license of the model is **independent** to that of the
dataset. That is to say, speech recognition models trained on the GigaSpeech dataset may be eligible for commercial license, 
provided they abide to the 'Fair Use' terms of the underlying data and do not violate any explicit copyright restrictions. 
This is likely to be true in most use-cases. However, it is your responsiblity to verify the appropriate model license for 
your specific use-case by confirming that the dataset usage abides by the Fair Use terms. SpeechColab is not responsible 
for the license of any machine learning model trained on the GigaSpeech dataset.

### Citation Information

Please cite this paper if you find this work useful:

```bibtext
@inproceedings{gigaspeech2,
  title={GigaSpeech 2: An Evolving, Large-Scale and Multi-domain ASR Corpus for Low-Resource Languages with Automated Crawling, Transcription and Refinement},
  author={Yifan Yang and Zheshu Song and Jianheng Zhuo and Mingyu Cui and Jinpeng Li and Bo Yang and Yexing Du and Ziyang Ma and Xunying Liu and Ziyuan Wang and Ke Li and Shuai Fan and Kai Yu and Wei-Qiang Zhang and Guoguo Chen and Xie Chen},
  booktitle={Proc. ACL},
  year={2025},
  address={Vienna},
}
```

## Terms of Access
The "Researcher" has requested permission to use the GigaSpeech 2 database (the "Database") 
at Tsinghua University. In exchange for such permission, Researcher hereby agrees to the 
following terms and conditions:

1. Researcher shall use the Database only for non-commercial research and educational purposes.
2. The SpeechColab team and Tsinghua University make no representations or warranties regarding the Database, including but not limited to warranties of non-infringement or fitness for a particular purpose.
3. Researcher accepts full responsibility for his or her use of the Database and shall defend and indemnify the SpeechColab team and Tsinghua University, including their employees, Trustees, officers and agents, against any and all claims arising from Researcher's use of the Database, including but not limited to Researcher's use of any copies of copyrighted audio files that he or she may create from the Database.
4. Researcher may provide research associates and colleagues with access to the Database provided that they first agree to be bound by these terms and conditions.
5. The SpeechColab team and Tsinghua University reserve the right to terminate Researcher's access to the Database at any time.
6. If Researcher is employed by a for-profit, commercial entity, Researcher's employer shall also be bound by these terms and conditions, and Researcher hereby represents that he or she is fully authorized to enter into this agreement on behalf of such employer.