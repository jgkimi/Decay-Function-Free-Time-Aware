# Decay-Function-Free Time-Aware Attention for SLU
This code is for the paper "Decay-Function-Free Time-Aware Attention to Context and Speaker Indicator for Spoken Language Understanding" accepted at NAACL 2019.

##
Requirement:
python>=2.7
tensorflow>=1.4.1
scikit-learn

## Data Preparation
We refer to https://github.com/MiuLab/Time-Decay-SLU (MiuLab repo).
1. Put the dstc4 data on "./dstc4/" (Check the path in ``parse_history.py``) and run ``parse_history.py`` to preprocess the data. (Then the preprocessed one will be located at "./Data".)
2. Download GloVe, modify line 33 in ``slu_preprocess.py`` to indicate the location of the glove file and run the code.

## Run
```
    python slu.py \
    --target [ALL, Guide, Tourist]
    --level [sentence, role]
    --talker_applied_to [Dist, Intent]
    --att_to [Dist, Intent]
    --att_out [Dist, Intent]
```

## Reference
If you found this code useful, please cite the paper:
```
@inproceedings{kim-lee-2019-decay,
    title = "Decay-Function-Free Time-Aware Attention to Context and Speaker Indicator for Spoken Language Understanding",
    author = "Kim, Jonggu and Lee, Jong-Hyeok",
    booktitle = "Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)",
    month = jun,
    year = "2019",
    address = "Minneapolis, Minnesota",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/N19-1372",
    pages = "3718--3726",
}
```

## Acknowledgement
Our implementation utilizes the code from MiuLab repo (https://github.com/MiuLab/Time-Decay-SLU).
