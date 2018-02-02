# coding=utf-8
import time
import logging
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import datetime
import os

import numpy as np

from polymerization import Polymerization
from data_helper import load_data, load_label, build_vocab, load_embedding, batch_iter, format_input_x, split_train_by_ratio, offline_test

#------------------------- define parameter -----------------------------
tf.flags.DEFINE_string("train_file", "../../data/context_train_5w.txt", "train corpus file")
tf.flags.DEFINE_string("test_file", "/export/lbc/corpus/commModel/data/test_corpus.txt", "test corpus file")
tf.flags.DEFINE_string("word_file", "../../context_data/words.txt", "test corpus file")
tf.flags.DEFINE_string("embedding_file", "../../context_data/vectors.txt", "vector file")
tf.flags.DEFINE_string("label_file", "../../data/labels.txt", "label file")
tf.flags.DEFINE_integer("rnn_size", 200, "rnn size of lstm")
tf.flags.DEFINE_integer("num_rnn_layers", 1, "the number of rnn layer")
tf.flags.DEFINE_integer("embedding_size", 150, "embedding size")
tf.flags.DEFINE_integer("attention_dim", 100, "embedding size")
tf.flags.DEFINE_integer("num_unroll_steps", 60, "sentence size")
tf.flags.DEFINE_string("filter_sizes", "1,2,3,4,5", "filter size of cnn")
tf.flags.DEFINE_integer("num_filters", 128, "the number of filter in every layer")
tf.flags.DEFINE_float("dropout", 0.5, "the proportion of dropout")
tf.flags.DEFINE_float("max_grad_norm", 5, "the max of gradient")
tf.flags.DEFINE_float("init_scale", 0.1, "initializer scale")
tf.flags.DEFINE_integer("batch_size", 128, "batch size of each batch")
tf.flags.DEFINE_float('lr',5e-4,'the learning rate')
tf.flags.DEFINE_float('lr_decay',0.6,'the learning rate decay')
tf.flags.DEFINE_integer("epoches", 100, "epoches")
tf.flags.DEFINE_integer('max_decay_epoch',30,'num epoch')
tf.flags.DEFINE_integer("evaluate_every", 1000, "run evaluation")
tf.flags.DEFINE_integer("checkpoint_every", 3000, "run evaluation")
tf.flags.DEFINE_integer("l2_reg_lambda", 1e-4, "l2 regulation")
tf.flags.DEFINE_string("out_dir", "save/", "output directory")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", True, "Log placement of ops on devices")
tf.flags.DEFINE_float("gpu_options", 0.1, "use memory rate")

FLAGS = tf.flags.FLAGS
filter_sizes = [int(filter_size) for filter_size in (FLAGS.filter_sizes).split(",")]
#----------------------------- define parameter end ---------------------------------- 
#----------------------------- define a logger ---------------------------------------
logger = logging.getLogger("execute")
logger.setLevel(logging.INFO)

fh = logging.FileHandler("./test.log", mode="w")
fh.setLevel(logging.INFO)

fmt = "%(asctime)-15s %(levelname)s %(filename)s %(lineno)d %(process)d %(message)s"
datefmt = "%a %d %b %Y %H:%M:%S"
formatter = logging.Formatter(fmt, datefmt)

fh.setFormatter(formatter)
logger.addHandler(fh)
#----------------------------- define a logger end -----------------------------------

def cal_detail_acc(prediction, correct_num, input_label, batch_corpus):
    otherId = label2id.get("other")
    input_other = [otherId] * len(input_label)
    other_pred = 0
    for idx in np.arange(len(prediction)):
        logger.info("\t" + id2label[prediction[idx]])
        if prediction[idx] == input_label[idx]:
            if int(prediction[idx]) == int(otherId):
                other_pred += 1
    total_other = np.sum(np.equal(input_label, input_other))
    return other_pred, total_other, int(correct_num - other_pred), int(len(input_label) - total_other)

#------------------------------- evaluate model -----------------------------------
def evaluate(session, test_x, test_y, corpus, global_steps=None):
    total_correct_num=0
    total_busi_num = 0
    total_busi_correct_num = 0
    total_other_num = 0
    total_other_correct_num = 0
    data = zip(test_x, test_y, corpus)
    total_num=len(data)
    for step, batch in enumerate(batch_iter(data, batch_size=FLAGS.batch_size, shuffle=False)):
        x, input_y, batch_corpus = zip(*batch)
        input_x, sess_len, sent_len = format_input_x(x)
        fetches = [model_prediction, model_correct_num, model_accuracy]
        feed_dict={
            model.input_data:input_x,
            model.target:input_y,
            model.session_lengths:sess_len,
            model.sent_lengths:sent_len,
            model.dropout_ratio:1.0
        }
        
        prediction, correct_num, acc = session.run(fetches, feed_dict)
        other_correct_num, other_num, busi_correct_num, busi_num = cal_detail_acc(prediction, correct_num, input_y, batch_corpus)
        total_correct_num += correct_num
        total_busi_num += busi_num
        total_busi_correct_num += busi_correct_num
        total_other_num += other_num
        total_other_correct_num += other_correct_num

    accuracy=float(total_correct_num)/total_num
    busi_acc = float(total_busi_correct_num) / total_busi_num
    other_acc = float(total_other_correct_num) / total_other_num
    logger.info("validation success")

    return accuracy, busi_acc, other_acc
#------------------------------ evaluate model end -------------------------------------

#------------------------------------load data -------------------------------
label2id, id2label = load_label(FLAGS.label_file)
id2word, word2id = build_vocab(FLAGS.word_file)
embeddings = load_embedding(FLAGS.embedding_file)
logger.info("load label, word, embedding finished")
train_valid_x, train_valid_y, class_weight = load_data(FLAGS.train_file, word2id, label2id, FLAGS.num_unroll_steps)

# cal label weight
class_weight_mean = np.mean(class_weight.values())
label_weight = {}
for label, weight in class_weight.items():
    label_weight[label2id.get(label)] = 1. / (weight / class_weight_mean)
label_weight = np.array([label_weight[ix] for ix in sorted(label_weight.keys())], dtype=np.float32)

# split data
train_data, valid_data = split_train_by_ratio(zip(train_valid_x, train_valid_y), 0.01)
train_x, train_y = zip(*train_data)
valid_x, valid_y = zip(*valid_data)
logger.info("load train data finish")
num_classes = len(label2id)
#test_x, test_y, _ = load_data(FLAGS.test_file, word2id, label2id, FLAGS.num_unroll_steps)
test_x, test_y, corpus = offline_test(FLAGS.test_file, word2id, label2id, FLAGS.num_unroll_steps)
logger.info("load test data finish")
#----------------------------------- load data end ----------------------

#----------------------------------- execute train ---------------------------------------
with tf.Graph().as_default():
    with tf.device("/cpu:3"):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_options)
        session_conf = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement, log_device_placement=FLAGS.log_device_placement, gpu_options=gpu_options)
        with tf.Session(config=session_conf).as_default() as sess:
            cell = rnn_cell.GRUCell(FLAGS.rnn_size)

            model = Polymerization(cell, embeddings, FLAGS.embedding_size, FLAGS.attention_dim, FLAGS.rnn_size, FLAGS.num_rnn_layers, num_classes, FLAGS.num_unroll_steps, label_weight, FLAGS.max_grad_norm, FLAGS.lr, FLAGS.num_filters, filter_sizes)
            model_logits, model_l2_loss = model.inference(FLAGS.dropout, False)
            model_cost, model_prediction, model_correct_num, model_accuracy = model.loss_and_acc(model_logits, model_l2_loss)
            model_train_op, model_summary = model.grad(model_cost, model_accuracy)

            #add summary
            train_summary_dir = os.path.join(FLAGS.out_dir,"summaries","train")
            train_summary_writer = tf.train.SummaryWriter(train_summary_dir,sess.graph)

            #add checkpoint
            saver = tf.train.Saver(tf.all_variables())
            checkpoint_dir = os.path.abspath(os.path.join(FLAGS.out_dir, "checkpoints"))
            #checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            #if not os.path.exists(checkpoint_dir):
            #    os.makedirs(checkpoint_dir)
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

            #tf.initialize_all_variables().run()
            global_steps=1
            test_accuracy, test_busi_acc, test_other_acc = evaluate(sess, test_x, test_y, corpus)
            logger.info("the test data accuracy is %6.7f, business accuracy is %6.7f, other accuracy is %6.7f"%(test_accuracy, test_busi_acc, test_other_acc))
#----------------------------------- execute train end -----------------------------------
