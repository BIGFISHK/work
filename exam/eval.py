# -*- coding: UTF-8 -*-

from train import *

test_file = conf.get('train_config', 'test_file')
test_InputExamples = get_data(os.path.join(input_path, test_file))
test_features = convert_examples_to_features(test_InputExamples, label_list, max_seq_length, tokenizer)

# 测试input函数
eval_input_fn = input_fn_builder(
    features=test_features,
    seq_length=max_seq_length,
    is_training=False,
    drop_remainder=False)


if __name__ == '__main__':
    evaluate_info = estimator.evaluate(input_fn=eval_input_fn, steps=None)
    print(evaluate_info)