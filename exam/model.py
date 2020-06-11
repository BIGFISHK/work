# -*- coding: UTF-8 -*-

from read_data import *
from bert import modeling, optimization
import configparser
import os

conf = configparser.ConfigParser()
conf.read('config.ini', encoding="utf-8-sig")
bert_config_file = conf.get('path', 'bert_config_file')
vocab_file = conf.get('path', 'vocab_file')
init_checkpoint = conf.get('path', 'init_checkpoint')
label_list = conf.get('train_config', 'label_list').split(',')
input_path = conf.get('path', 'input_dir')
output_path = conf.get('path', 'output_dir')
batch_size = int(conf.get('train_config', 'batch_size'))
learning_rate = float(conf.get('train_config', 'learning_rate'))
num_train_epochs = int(conf.get('train_config', 'num_train_epochs'))
max_seq_length = int(conf.get('train_config', 'max_seq_length'))
warm_up_proportion = float(conf.get('train_config', 'warm_up_proportion'))
save_checkpoints_steps = int(conf.get('train_config', 'save_checkpoints_steps'))
save_summary_steps = int(conf.get('train_config', 'save_summary_steps'))
train_file = conf.get('train_config', 'train_file')

bert_config = modeling.BertConfig.from_json_file(bert_config_file)
train_InputExamples = get_data(os.path.join(input_path, train_file))
tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=False)
train_features = convert_examples_to_features(train_InputExamples, label_list, max_seq_length, tokenizer)


def create_model(config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings):
    model = modeling.BertModel(
        config=config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    output_layer = model.get_pooled_output()
    hidden_size = output_layer.shape[-1].value

    output_weights = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable("output_bias", [num_labels], initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):
        if is_training:
            # 0.1比例的 dropout
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        # 计算 softmax 值
        probabilities = tf.nn.softmax(logits, axis=-1)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        # 计算loss
        loss = tf.reduce_mean(per_example_loss)
        return (loss, per_example_loss, logits, probabilities)


# 返回一个 `model_fn`函数闭包给 Estimator
def model_fn_builder(config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps):
    #构建模型
    def model_fn(features, labels, mode, params):

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]
        is_real_example = None
        if "is_real_example" in features:
            is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
        else:
            is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        use_one_hot_embeddings = False

        # 使用参数构建模型,input_idx 就是输入的样本idx表示，label_ids 就是标签的idx表示
        (total_loss, per_example_loss, logits, probabilities) = create_model(
            config, is_training, input_ids, input_mask, segment_ids, label_ids,
            num_labels, use_one_hot_embeddings)

        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        scaffold_fn = None
        # 加载BERT预训练模型
        if init_checkpoint:
            (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape, init_string)

        output_spec = None
        # 训练模式
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu=False)

            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op)
        # 评估模式
        elif mode == tf.estimator.ModeKeys.EVAL:
            def metric_fn(per_example_loss, label_ids, logits, is_real_example):
                predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
                accuracy = tf.metrics.accuracy(
                    labels=label_ids, predictions=predictions, weights=is_real_example)
                loss = tf.metrics.mean(values=per_example_loss, weights=is_real_example)
                return {
                    "eval_accuracy": accuracy,
                    "eval_loss": loss,
                }

            eval_metrics = metric_fn(per_example_loss, label_ids, logits, is_real_example)
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metric_ops=eval_metrics)
        # 测试模式
        else:
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                predictions={"probabilities": probabilities})
        return output_spec
    return model_fn


# 计算训练步数和预热步数
num_train_steps = int(len(train_features) / batch_size * num_train_epochs)
num_warmup_steps = int(num_train_steps * warm_up_proportion)

model_fn = model_fn_builder(
    config=bert_config,
    num_labels=len(label_list),
    learning_rate=learning_rate,
    init_checkpoint=init_checkpoint,
    num_train_steps=num_train_steps,
    num_warmup_steps=num_warmup_steps)