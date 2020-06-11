# -*- coding: UTF-8 -*-

from model import *

tf.logging.set_verbosity(tf.logging.INFO)


# 创建一个input_fn供 Estimator 使用
def input_fn_builder(features, seq_length, is_training, drop_remainder):

    all_input_ids = []
    all_input_mask = []
    all_segment_ids = []
    all_label_ids = []

    for feature in features:
        all_input_ids.append(feature.input_ids)
        all_input_mask.append(feature.input_mask)
        all_segment_ids.append(feature.segment_ids)
        all_label_ids.append(feature.label_id)

    # 正真的 input 方法
    def input_fn(params):
        batch = params["batch_size"]
        num_examples = len(features)
        d = tf.data.Dataset.from_tensor_slices({
            "input_ids":
                tf.constant(
                    all_input_ids, shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "input_mask":
                tf.constant(
                    all_input_mask,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "segment_ids":
                tf.constant(
                    all_segment_ids,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "label_ids":
                tf.constant(all_label_ids, shape=[num_examples], dtype=tf.int32),
        })

        # 训练模式需要shuffling
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.batch(batch_size=batch, drop_remainder=drop_remainder)
        return d
    return input_fn


# 生成 tf.estimator 运行配置
run_config = tf.estimator.RunConfig(
    model_dir=output_path,
    save_summary_steps=save_summary_steps,
    save_checkpoints_steps=save_checkpoints_steps)

# 模型estimator
estimator = tf.estimator.Estimator(
    model_fn=model_fn,
    config=run_config,
    params={"batch_size": batch_size})

# 训练input函数
train_input_fn = input_fn_builder(
    features=train_features,
    seq_length=max_seq_length,
    is_training=True,
    drop_remainder=False)  # 是否丢弃batch剩余样本


if __name__ == '__main__':
    print("开始训练")
    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
    print("训练结束")
