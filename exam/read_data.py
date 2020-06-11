# -*- coding: UTF-8 -*-

import pandas as pd
import tensorflow as tf
from bert import tokenization


class InputExample(object):
    def __init__(self, text_a, label=None):
        self.text_a = text_a
        self.label = label


class InputFeatures(object):
    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 label_id,
                 is_real_example=True):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.is_real_example = is_real_example


class PaddingInputExample(object):
    pass


def get_data(path):
    data = pd.read_csv(path, encoding='gbk')
    input_examples = data.apply(lambda x: InputExample(text_a=x['TEXT'], label=x['LABEL']), axis=1)
    return input_examples


def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer):
    # 将单个的 InputExample 转换成单个的 InputFeatures
    if isinstance(example, PaddingInputExample):
        return InputFeatures(
            input_ids=[0] * max_seq_length,
            input_mask=[0] * max_seq_length,
            segment_ids=[0] * max_seq_length,
            label_id=0,
            is_real_example=False)

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    tokens_a = tokenizer.tokenize(example.text_a)

    # "- 2" 是因为[CLS] 和 [SEP]两个符号
    if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[0 - (max_seq_length - 2):]

    tokens = []
    segment_ids = []
    tokens.append("[CLS]")  # 句头添加 [CLS] 标志
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")  # 句尾添加[SEP] 标志
    segment_ids.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)

    # 用0填充序列空余位置
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    label_id = label_map[example.label]

    # 结构化为一个类
    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_id=label_id,
        is_real_example=True)
    return feature


# 将InputExample转换为InputFeature
def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))
        feature = convert_single_example(ex_index, example, label_list,
                                         max_seq_length, tokenizer)
        features.append(feature)
    return features


