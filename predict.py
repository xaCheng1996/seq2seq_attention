import tensorflow as tf
import data_load
from model import Seq2SeqModel
from nltk import bleu
import sys
import numpy as np


tf.app.flags.DEFINE_integer('rnn_size',1024, 'Number of hidden units in each layer')
tf.app.flags.DEFINE_integer('num_layers', 2, 'Number of layers in each encoder and decoder')
tf.app.flags.DEFINE_integer('embedding_size', 100, 'Embedding dimensions of encoder and decoder inputs')

tf.app.flags.DEFINE_float('learning_rate', 0.0001, 'Learning rate')
tf.app.flags.DEFINE_integer('batch_size',128, 'Batch size')
tf.app.flags.DEFINE_integer('numEpochs', 30, 'Maximum # of training epochs')
tf.app.flags.DEFINE_integer('steps_per_checkpoint', 100, 'Save model checkpoint every this iteration')
tf.app.flags.DEFINE_string('model_dir', 'model/', 'Path to save model checkpoints')
tf.app.flags.DEFINE_string('model_name', 'chatbot.ckpt', 'File name used for model checkpoints')
FLAGS = tf.app.flags.FLAGS

data_path = '/home/share/chengxuanang/data_origin/test/x.txt'
data_tar = '/home/share/chengxuanang/data_origin/test/y.txt'
inputs, target, word2id, id2word, vocab_size = data_load.loadDataset(data_path, data_tar)
final_target = []
embedding_matrix = data_load.embedding_re()


def predict_ids_to_seq(predict_ids, id2word, beam_szie, batch_size):
    '''
    将beam_search返回的结果转化为字符串
    :param predict_ids: 列表，长度为batch_size，每个元素都是decode_len*beam_size的数组
    :param id2word: vocab字典
    :return:
    '''
    for single_predict in predict_ids:
        for i in range(beam_szie):
           # print(single_predict)
           # print(np.array(single_predict).shape)
            predict_list = np.ndarray.tolist(single_predict[:,:, i])
            for pred in predict_list:
                predict_seq = [id2word[idx] for idx in pred if idx >= 0]
                final_target.append(predict_seq)
with tf.Session() as sess:
    model = Seq2SeqModel(FLAGS.rnn_size, FLAGS.num_layers, FLAGS.embedding_size, FLAGS.learning_rate, word2id,
                         embedding_matrix=embedding_matrix,
                         mode='decode', use_attention=True, beam_search=True, beam_size=1, max_gradient_norm=5.0)
    ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
    saver = tf.train.Saver(tf.global_variables())
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print('Reloading model parameters..')
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        raise ValueError('No such file:[{}]'.format(FLAGS.model_dir))
    batches = data_load.getBatches(inputs, target, 128)
    for batch in batches:
        # 获得预测的id
        x_text = list(data_load.id_transfer_word(batch.encoder_inputs))
        y_text = list(data_load.id_transfer_word(batch.decoder_targets))
        predicted_ids= model.infer(sess, batch)
        #print(embedd)
        # print(predicted_ids)
        # 将预测的id转换成汉字
        final_target = []
        file = "./eval_output.txt"
        output = open(file, "a")
        predict_ids_to_seq(predicted_ids, id2word, 1, len(batch.encoder_inputs))
        for k in range(len(final_target)):
            print(x_text[k][::-1])
            print(y_text[k])
            #print(final_target[k])
            str1 = ""
            str2 = ""
            for i in y_text[k]:
                if i is not "<PAD>":
                    str2 = str2 + str(i)
            for i in final_target[k]:
                str1 = str1 + str(i)
            print(str1)
            print("\n")
            num = bleu(str1, str2)
            output.write(str(x_text[k][::-1]))
            output.write("\n")
            output.write(str2)
            output.write("\n")
            output.write(str1)
            output.write("\n")
            output.write(str(num))
            output.write("\n")
            output.write("\n")
