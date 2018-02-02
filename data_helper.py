#!/usr/bin/python
# -*- coding:utf-8 -*-
import codecs
import numpy as np
import copy

blackCatg = ["1", u"多问"]

def parseCorpus(filename, corpus):
    with codecs.open(filename, mode="r", encoding="utf-8") as rf:
        sessTalk = list()
        busiCatg = ""
        prevCatg = list()
        isSess = False
        for line in rf.readlines():
            arr = line.split("\t")
            if len(arr) == 5 and len(arr[2].strip()) == 0 and len(arr[3].strip()) == 0:
                if len(sessTalk) == 0:
                    pass
                elif len(sessTalk) >= 2 and len(sessTalk) <= 20 and busiCatg != "" and isSess == True:
                    sessTalk.append(line)
                    corpus.append(sessTalk)
                    sessTalk = list()
                    busiCatg = ""
                    prevCatg = list()
                    isSess = False
                else:
                    sessTalk = list()
                    busiCatg = ""
                    prevCatg = list()
                    isSess = False
            elif len(arr) == 5 and (len(arr[1].strip()) == 0 or arr[1].strip() not in blackCatg) and arr[2].strip() not in blackCatg:
               sessTalk.append(line)
               if arr[2].strip() != "other":
                   busiCatg = arr[2].strip()
               if arr[1].strip() in prevCatg:
                   isSess = True
               prevCatg.append(arr[2].strip())
            elif len(arr) == 5 and (arr[1].strip() in blackCatg or arr[2].strip() in blackCatg):
               pass
            else:
               sessTask = list()
               busiCatg = ""
               prevCatg = list()
               isSess = False

def split_train_test(corpus, ratio):
    testSessNum = int(np.ceil(len(corpus) * ratio))
    shuffle_idx = np.random.permutation(np.arange(len(corpus)))
    shuffle_corpus = [corpus[idx] for idx in shuffle_idx]
    testCorpus = shuffle_corpus[:(testSessNum + 1)]
    trainCorpus = shuffle_corpus[(testSessNum + 1):]
    return testCorpus, trainCorpus

def save2file(corpus, tofile):
    with codecs.open(tofile, mode="w", encoding="utf-8") as wf:
        for sessTalk in corpus:
            for line in sessTalk:
                wf.write(line)

def save2File(corpus, catgs, tofile, delimiter="\t"):
    with codecs.open(tofile, mode="w", encoding="utf-8") as wf:
        ix = 0
        for sessTalk in corpus:
            for line in sessTalk:
                wf.write(line)
                wf.write(delimiter)
            wf.write(catgs[ix])
            wf.write("\n")
            ix += 1

def cutword(context):
    if len(context.strip()) == 0:
        return None

    sentence = ""
    phrases = context.split("\t")[0].split(" ")
    for phrase in phrases:
        flag = True
        for word in phrase:
            if 'A' <= word <= 'Z' or word == '_':
                continue
            else:
                flag = False
                break
        if flag:
            sentence += phrase + " "
        else:
            for word in phrase:
                sentence += word + " "

    return sentence

def parseSessData(filename):
    sessTalks = list()
    catgs = list()
    with codecs.open(filename, mode="r", encoding="utf-8") as rf:
        sessTalk = list()
        for line in rf.readlines():
            arr = line.split("\n")[0].split("\t")
            if len(arr) == 5:
                if len(arr[-1].strip()) == 0 and len(sessTalk) != 0:
                    sessTalk = list()
                else:
                    genQuest = cutword(arr[-1].strip())
                    catg = arr[2].strip()
                    sessTalk.append(genQuest)
                    if len(sessTalk) > 3:
                        tmpTalk = sessTalk[len(sessTalk) - 3:]
                    else:
                        tmpTalk = sessTalk
                    sessTalks.append(copy.deepcopy(tmpTalk))
                    catgs.append(catg)

    return sessTalks, catgs

def load_embedding(filename):
    embeddings = []
    with codecs.open(filename, "r", encoding="utf-8") as rf:
        for line in rf.readlines():
            embeddings.append([float(val) for val in line.strip().split(" ")])

    embeddings = np.array(embeddings)
    embedding_size = embeddings.shape[1]
    unknown_padding_embedding = np.random.normal(0, 0.1, (2, embedding_size))
    
    embeddings = np.append(np.array(embeddings, dtype=np.float32), unknown_padding_embedding.astype(np.float32), axis=0)
    return embeddings

def build_vocab(filename):
    id2word = []
    with codecs.open(filename, "r", encoding="utf-8") as rf:
        for line in rf.readlines():
            arr = line.split(" ")
            id2word.append(arr[0].strip())

    id2word.append("UNKNOWN")
    id2word.append("<a>")
    word2id = {x : i for i, x in enumerate(id2word)}

    return id2word, word2id

def load_label(filename):
    id2label = []
    with codecs.open(filename, "r", encoding="utf-8") as rf:
        for line in rf.readlines():
            arr = line.split("\n")[0].split("\t")
            id2label.append(arr[0])

    label2id = {x : i for i, x in enumerate(id2label)}

    return label2id, id2label

# [[1,2,3],[2,3,4]]
def sent2ix(word2ix, label2id, arr, sent_size):
    sentixs = list()
    for sent in arr:
        sentix = list()
        words = sent.split(" ")
        for word in words:
            sentix.append(word2ix.get(word, word2ix.get("UNKNOWN")))

        if len(sentix) < sent_size:
            padix = [word2ix.get("<a>")] * (sent_size - len(sentix))
            sentix.extend(padix)
        elif len(sentix) > sent_size:
            sentix = sentix[:sent_size]
        else:
            pass

        sentixs.append(sentix)
    return sentixs

def load_data(corpusFile, word2id, label2id, sent_size):
    sents, labels = [], []
    label2sents = {}
    with codecs.open(corpusFile, "r", "utf-8") as rf:
        for line in rf.readlines():
            arr = line.split("\n")[0].split("\t")
            label = arr[-1].strip()
            if label not in label2id:
                continue

            labelId = label2id.get(label)
            
            sents.append(sent2ix(word2id, label2id, arr[:-1], sent_size))
            labels.append(labelId)

            if label not in label2sents:
                label2sents[label] = 1
            else:
                label2sents[label] += 1

    return sents, labels, label2sents 

def format_input_x(inputs):
    batch_size = len(inputs)

    session_sizes = np.array([len(sess) for sess in inputs], dtype=np.int32)
    session_size = session_sizes.max()

    sentence_sizes_ = [[len(sent) for sent in sess] for sess in inputs]
    sentence_size = max(map(max, sentence_sizes_))#max length of sentence

    b = np.zeros(shape=[batch_size, session_size, sentence_size], dtype=np.int32) # == PAD

    sentence_sizes = np.zeros(shape=[batch_size, session_size], dtype=np.int32)
    for i, session in enumerate(inputs):
        for j, sentence in enumerate(session):
            sentence_sizes[i, j] = sentence_sizes_[i][j]
            for k, word in enumerate(sentence):
                b[i, j, k] = word

    return b, session_sizes, sentence_sizes

def batch_iter(data, batch_size, shuffle=True):
    """
    iterate the data
    """
    data_len = len(data)
    batch_num = int(np.ceil(data_len / batch_size))
    data = np.array(data)

    if shuffle:
        shuffle_idx = np.random.permutation(np.arange(data_len))
        shuffle_data = data[shuffle_idx]
    else:
        shuffle_data = data

    for batch in range(batch_num):
        start_idx = batch * batch_size
        end_idx = min((batch + 1) * batch_size, data_len)
        yield shuffle_data[start_idx : end_idx]

def split_train_by_ratio(corpus, ratio):
    data_len = len(corpus)
    valid_num = int(np.ceil(data_len * ratio))
    shuffle_idx = np.random.permutation(np.arange(data_len))
    shuffle_data = np.array(corpus)[shuffle_idx]
    valid_data = shuffle_data[:valid_num]
    train_data = shuffle_data[valid_num :]
    return train_data, valid_data

def filter_other(fromfile, tofile, limit):
    busi_corpus = []
    other_corpus = []
    with codecs.open(fromfile, "r") as rf:
        for line in rf.readlines():
            catg = line.split("\n")[0].split("\t")[-1]
            if "other" == catg:
                if line not in other_corpus:
                    other_corpus.append(line)
            else:
                if line not in busi_corpus:
                    busi_corpus.append(line)

    shuffleIx = np.random.permutation(np.arange(len(other_corpus)))
    other_corpus = [other_corpus[ix] for ix in shuffleIx]
    other_corpus = other_corpus[:limit]
    busi_corpus.extend(other_corpus)
    shuffleIdx = np.random.permutation(np.arange(len(busi_corpus)))
    corpus = [busi_corpus[ix] for ix in shuffleIdx]
    
    with codecs.open(tofile, "w") as wf:
        for line in corpus:
            wf.write(line)

def offline_test(filename, word2id, label2id, sent_size):
    test_x, test_y, corpus = [], [], []
    with codecs.open(filename, "r", "utf-8") as rf:
        for line in rf.readlines():
            line = line.split("\n")[0]
            arr = line.split("\t")
            label = arr[-1].strip()
            labelId = label2id.get(label)
            if labelId is None:
                labelId = 1

            sent = sent2ix(word2id, label2id, arr[:-1], sent_size)

            test_x.append(sent)
            test_y.append(labelId)
            corpus.append(line)
    return test_x, test_y, corpus

def main():
    trainFile = "trian.txt"
    testFile = "test.txt"
    badcaseFile = "badCase.txt"
    corpusFile = "commModel_corpus.txt"
    testCorpusFile = "commModel_test_corpus.txt"
    trainCorpusFile = "commModel_train_corpus.txt"
    if False:
        corpus = list()
        parseCorpus(trainFile, corpus)
        parseCorpus(testFile, corpus)
        parseCorpus(badcaseFile, corpus)

        testCorpus, trainCorpus = split_train_test(corpus, ratio=0.1)

        save2file(corpus, corpusFile)
        save2file(testCorpus, testCorpusFile)
        save2file(trainCorpus, trainCorpusFile)

    formatTrainCorpus = "../../context_data/commModel_format_train.txt"
    formatTestCorpus = "../../context_data/commModel_format_test.txt"
    if False:
        trainData, trainCatgs = parseSessData(trainCorpusFile)
        testData, testCatgs = parseSessData(testCorpusFile)
        save2File(trainData, trainCatgs, formatTrainCorpus)
        save2File(testData, testCatgs, formatTestCorpus)

    if True:
        filter_train = "../../context_data/commModel_format_train_30000.txt"
        filter_test = "../../context_data/commModel_format_test_30000.txt"
        filter_other(formatTrainCorpus, filter_train, 30000)
        filter_other(formatTestCorpus, filter_test, 30000)
       

if __name__ == "__main__":
    main()
