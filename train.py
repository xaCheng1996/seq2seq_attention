import tensorflow as tf
import data_helpers
from model import Seq2SeqModel
import math
import data_load
import os

tf.app.flags.DEFINE_integer('rnn_size', 1024, 'Number of hidden units in each layer')
tf.app.flags.DEFINE_integer('num_layers', 2, 'Number of layers in each encoder and decoder')
tf.app.flags.DEFINE_integer('embedding_size', 100, 'Embedding dimensions of encoder and decoder inputs')

tf.app.flags.DEFINE_float('learning_rate', 0.0001, 'Learning rate')
tf.app.flags.DEFINE_integer('batch_size',128, 'Batch size')
tf.app.flags.DEFINE_integer('numEpochs', 30, 'Maximum # of training epochs')
tf.app.flags.DEFINE_integer('steps_per_checkpoint',100, 'Save model checkpoint every this iteration')
tf.app.flags.DEFINE_string('model_dir', 'model/', 'Path to save model checkpoints')
tf.app.flags.DEFINE_string('model_name', 'description.ckpt', 'File name used for model checkpoints')
FLAGS = tf.app.flags.FLAGS

# """
# 数据路径，根据自己的状态修改
# """
data_path = '/data_origin/train/x.txt'
data_tar = '/data_origin/train/y.txt'
data_path_val = "/data_origin/val/x.txt"
data_tar_val = '/data_origin/val/y.txt'

# """
# data.load.loadDataset加载word2vec矩阵，将数据转化为vector输入
# """
inputs, target, word2id, id2word, vocab_size = data_load.loadDataset(data_path, data_tar)
inputs_val, target_val, word2id_val, id2word_val, vocab_size_val = data_load.loadDataset(data_path_val, data_tar_val)

# """
# 加载词向量矩阵
# """
embedding_matrix = data_helpers.embedding_re()

with tf.Session() as sess:
​    model = Seq2SeqModel(FLAGS.rnn_size,
​                         FLAGS.num_layers,
​                         FLAGS.embedding_size,
​                         FLAGS.learning_rate,
​                         word2id,
​                         embedding_matrix=embedding_matrix,
​                         mode='train',
​                         use_attention=True,
​                         beam_search=True,
​                         beam_size=5,
​                         max_gradient_norm=5.0
​                         )
​    saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
​    loss_min = 500.0
​    # 如果存在已经保存的模型的话，就继续训练，否则，就重新开始
​    ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
​    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
​        print('Reloading model parameters..')
​        saver.restore(sess, ckpt.model_checkpoint_path)
​    else:
​        print('Created new model parameters..')
​        sess.run(tf.global_variables_initializer())

    """
    train+validation
    validation添加后模型效果反而下降了……不知道原因
    """
    current_step = 0
    summary_writer = tf.summary.FileWriter(FLAGS.model_dir, graph=sess.graph)
    for e in range(FLAGS.numEpochs):
        print("----- Epoch {}/{} -----".format(e + 1, FLAGS.numEpochs))
        batches = data_load.getBatches(inputs,target, FLAGS.batch_size)
        for nextBatch in batches:
            # print(nextBatch)
            loss, summary, pred = model.train(sess, nextBatch)
            current_step += 1
            # 每多少步进行一次保存
 #           if current_step % 5 == 0:
 #               if e == 0:
 #                   with open("./train_output.txt", "a", encoding="utf-8") as train_out:
 #                       x_text = list(data_load.id_transfer_word(nextBatch.encoder_inputs))
 #                       y_text = list(data_load.id_transfer_word(nextBatch.decoder_targets))
 #                       final = data_load.id_transfer_word(pred)
 #                       for k in range(len(final)):
 #                           str11 = ""
 #                           str2 = ""
 #                           for i in y_text[k]:
 #                               if i is not "<PAD>":
 #                                   str2 = str2 + str(i)
 #                           for i in final[k]:
 #                               str11 = str11 + str(i)
 #                           train_out.write(str(x_text[k][::-1]))
 #                           train_out.write("\n")
 #                           train_out.write(str2)
 #                           train_out.write("\n")
 #                           train_out.write(str11)
 #                           train_out.write("\n")
 #                           train_out.write("\n")
 #               print("now training loss in step %d: %.2f"%(current_step,loss))
            if current_step % FLAGS.steps_per_checkpoint == 0:
                batches_val = data_load.getBatches(inputs_val, target_val, FLAGS.batch_size)
                for batch_val in batches_val:
                    loss += model.eval(sess, batch_val)
                loss = loss/len(batches_val)
                perplexity = math.exp(float(loss)) if loss < 300 else float('inf')
                print("----- Step %d -- validation Loss %.2f -- Perplexity %.2f" % (current_step, loss, perplexity))
                if loss < loss_min:
                    print("loss is minimum, save the model")
                    summary_writer.add_summary(summary, current_step)
                    checkpoint_path = os.path.join(FLAGS.model_dir, FLAGS.model_name)
                    saver.save(sess, checkpoint_path, global_step=current_step)
                    loss_min = loss
                    print("now loss_minimum is %.3f"%loss_min)
                else:
                    print("the loss is too large, train continue")
                    print("now loss_minimum is %.3f"%loss_min)

