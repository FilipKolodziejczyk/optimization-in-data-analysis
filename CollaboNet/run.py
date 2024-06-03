import argparse
from collections import OrderedDict, defaultdict  # for OrderedDict()

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tensorflow as tf

from model.CollaboNet import *
from model.RunModel import *

if __name__ == '__main__':

    #Hyper Params
    parser = argparse.ArgumentParser()
    parser.add_argument('--guidee_data', type=str, help='data name', default='name')
    parser.add_argument('--pretrained', type=int, help='pretrained STM expName', default=0)
    parser.add_argument('--ncbi', action='store_true', help='include ncbi data', default=False)
    parser.add_argument('--jnlpba', action='store_true', help='include jnlpba data', default=False)
    parser.add_argument('--bc2', action='store_true', help='include bc2gm data', default=False)
    parser.add_argument('--bc4', action='store_true', help='include bc4chemd data', default=False)
    parser.add_argument('--bc5_disease', action='store_true', help='include bc5-disease data', default=False)
    parser.add_argument('--bc5_chem', action='store_true', help='include bc5-chem data', default=False)
    parser.add_argument('--bc5', action='store_true', help='include bc5cdr data', default=False)
    parser.add_argument('--tensorboard', action='store_true', help='single flag [default]False', default=False)
    parser.add_argument('--epoch', type=int, help='max epoch', default=100)
    parser.add_argument('--num_class', type=int, help='result class bio(3) [default]biolu(5)', default=5)
    parser.add_argument('--ce_dim', type=int, help='char embedding dim', default=30)
    parser.add_argument('--clwe_dim', type=int, help='char level word embedding dim', default=200)
    parser.add_argument('--clwe_method', type=str, help='clwe method: CNN biLSTM', default='CNN')
    parser.add_argument('--batch_size', type=int, help='batch size', default=10)
    parser.add_argument('--hidden_size', type=int, help='lstm hidden layer size', default=300)
    parser.add_argument('--lr', type=float, help='learning rate', default=0.01)
    parser.add_argument('--lr_decay', type=float, help='learning rate dacay rate', default=0.05)
    parser.add_argument('--lr_pump', action='store_true', help='do lr_pump', default=False)
    parser.add_argument('--loss_weight', type=float, help='loss weight between CRF, LSTM', default=1)
    parser.add_argument('--fc_method', type=str, help='fc method', default='normal')
    parser.add_argument('--mlp_layer', type=int, help='num highway layer ', default=1)
    parser.add_argument('--char_maxlen', type=int, help='char max length', default=49)
    parser.add_argument('--embdropout', type=float, help='input embedding dropout_rate', default=0.5)
    parser.add_argument('--lstmdropout', type=float, help='lstm output dropout_rate', default=0.3)
    parser.add_argument('--seed', type=int, help='seed value', default=0)
    parser.add_argument('--optimizer', type=str, help='Choice of optimizer',
                        choices=['adam', 'rmsprop', 'adagrad'], default='adagrad')
    args = parser.parse_args()

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #tf verbose off(info, warning)

    #seed initialize
    expName = setExpName()
    if args.seed != 0:
        seedV = int(args.seed % 100000)
    else:
        try:
            tempSeed = int(expName)
        except:
            tempSeed = int(expName[:12])
        seedV = int(tempSeed % 100000)

    random.seed(seedV)
    np.random.seed(seedV)
    tf.compat.v1.set_random_seed(seedV)

    #gpu setting
    gpu_config = tf.compat.v1.ConfigProto(device_count={'GPU': 1})  # only use GPU no.1
    gpu_config.gpu_options.allow_growth = True  # only use required resource(memory)
    gpu_config.gpu_options.per_process_gpu_memory_fraction = 1  # restrict to 100%

    ID2wordVecIdx, wordVec2LineNo, wordEmbedding = input_wordVec()
    ID2char = pickle.load(open('data/ID2char.pickle', 'rb'))

    m_train = 'train_dev'
    m_dev = 'dev'
    m_test = 'test'

    modelDict = OrderedDict()
    if args.ncbi:
        ncbi_args = deepcopy(args)
        ncbi_args.guidee_data = 'NCBI'
        modelDict['NCBI'] = {'args': ncbi_args}
    if args.jnlpba:
        jnl_args = deepcopy(args)
        jnl_args.guidee_data = 'JNLPBA'
        modelDict['JNLPBA'] = {'args': jnl_args}
    if args.bc2:
        bc2_args = deepcopy(args)
        bc2_args.guidee_data = 'BC2GM'
        modelDict['BC2GM'] = {'args': bc2_args}
    if args.bc4:
        bc4_args = deepcopy(args)
        bc4_args.guidee_data = 'BC4CHEMD'
        modelDict['BC4CHEMD'] = {'args': bc4_args}
    if args.bc5_chem:
        bc5c_args = deepcopy(args)
        bc5c_args.guidee_data = 'BC5CDR-chem'
        modelDict['BC5CDR-chem'] = {'args': bc5c_args}
    if args.bc5_disease:
        bc5d_args = deepcopy(args)
        bc5d_args.guidee_data = 'BC5CDR-disease'
        modelDict['BC5CDR-disease'] = {'args': bc5d_args}
    if args.bc5:
        bc5_args = deepcopy(args)
        bc5_args.guidee_data = 'BC5CDR'
        modelDict['BC5CDR'] = {'args': bc5_args}

    modelStart = time.time()
    modelClass = Model(args, wordEmbedding, seedV)

    for dataSet in modelDict:
        modelDict[dataSet]['summery'] = dict()
        modelDict[dataSet]['CLWE'] = modelClass.clwe(args=modelDict[dataSet]['args'], ID2char=ID2char)
        modelDict[dataSet]['WE'] = modelClass.we(args=modelDict[dataSet]['args'])
        modelDict[dataSet]['model'] = modelClass.model(args=modelDict[dataSet]['args'],
                                                       X_embedded_data=modelDict[dataSet]['WE'],
                                                       X_embedded_char=modelDict[dataSet]['CLWE'],
                                                       guideeInfo=None,
                                                       summery=modelDict[dataSet]['summery'],
                                                       scopename=dataSet)  #guideeInfo=None cuz in function we define
    dataNames = list()
    for dataSet in modelDict:
        modelDict[dataSet]['trainLossList'] = list()
        modelDict[dataSet]['testLossList'] = list()
        modelDict[dataSet]['trainDurationList'] = list()
        modelDict[dataSet]['testDurationList'] = list()
        modelDict[dataSet]['f1ValList'] = list()
        modelDict[dataSet]['f1ValWOCRFList'] = list()
        modelDict[dataSet]['maxF1'] = 0.0
        modelDict[dataSet]['maxF1idx'] = 0
        modelDict[dataSet]['prevF1'] = 0.0
        modelDict[dataSet]['stop_counter'] = 0
        modelDict[dataSet]['early_stop'] = False
        modelDict[dataSet]['m_name'] = modelDict[dataSet]['args'].guidee_data
        modelDict[dataSet]['precisionValList'] = list()
        modelDict[dataSet]['recallValList'] = list()

        dataNames.append(dataSet)

        try:
            os.mkdir('./modelSave/' + expName + '/' + modelDict[dataSet]['m_name'])
        except OSError as e:
            if e.errno == errno.EEXIST:  # if file exists! Python2.7 doesn't support file exist exception so need to use this
                print('./modelSave/' + expName + '/' + modelDict[dataSet]['m_name'] + ' Directory exists! not created.')
                suffix += 1
            else:
                raise

        modelDict[dataSet]['runner'] = RunModel(model=modelDict[dataSet]['model'], args=modelDict[dataSet]['args'],
                                                ID2wordVecIdx=ID2wordVecIdx, ID2char=ID2char,
                                                expName=expName, m_name=modelDict[dataSet]['m_name'], m_train=m_train,
                                                m_dev=m_dev, m_test=m_test)

    with tf.compat.v1.Session(config=gpu_config) as sess:
        phase = 0
        random.seed(seedV)
        np.random.seed(seedV)
        tf.compat.v1.set_random_seed(seedV)
        sess.run(tf.compat.v1.global_variables_initializer())
        saver = tf.compat.v1.train.Saver(tf.compat.v1.trainable_variables(), max_to_keep=10000)
        loader = tf.compat.v1.train.Saver(tf.compat.v1.trainable_variables(), max_to_keep=10000)

        print('====INITIALIZATION====')
        for epoch_idx in range(args.epoch * len(dataNames)):
            dataSet = dataNames[epoch_idx % len(dataNames)]
            if epoch_idx % len(dataNames) == 0:
                if args.pretrained != 0:
                    phase += 1
                print("[%d phase]" % (phase))
            m_name = modelDict[dataSet]['m_name']
            if modelDict[dataSet]['early_stop']:
                continue
            if modelDict[dataSet]['args'].tensorboard:
                tbWriter = tf.compat.v1.summary.FileWriter('./modelSave/' + expName + '/' + m_name + '/train', sess.graph)
            else:
                tbWriter = None

            print('====' + m_name.upper() + "_MODEL Training=====")
            startTime = time.time()
            batch_idx = random.sample(range(0, len(modelDict[dataSet]['runner'].m_batchgroup[m_train])),
                                      len(modelDict[dataSet]['runner'].m_batchgroup[m_train]))

            if args.pretrained == 0:
                intOuts = None
                early_stops = [24, 30, 30, 30, 25, 25]

            if args.pretrained != 0:
                intOuts = None
                early_stops = [5, 16, 23, 30, 10, 16]

            if ((epoch_idx / len(dataNames)) == 0) and (args.pretrained != 0):
                intOuts = dict()
                intOuts[m_train] = list()
                intOuts[m_dev] = list()
                intOuts[m_test] = list()
                for d_sub in modelDict:
                    if d_sub == dataSet:
                        continue
                    else:
                        loadpath = './modelSave/' + str(args.pretrained) + '/' + d_sub + '/'
                        loader.restore(sess, tf.train.latest_checkpoint(loadpath))

                        intOuts[m_train].append(
                            modelDict[d_sub]['runner'].info1epoch(m_train, modelDict[dataSet]['runner'], sess))
                        intOuts[m_dev].append(
                            modelDict[d_sub]['runner'].info1epoch(m_dev, modelDict[dataSet]['runner'], sess))
                        intOuts[m_test].append(
                            modelDict[d_sub]['runner'].info1epoch(m_test, modelDict[dataSet]['runner'], sess))

                loadpath = './modelSave/' + str(args.pretrained) + '/' + dataSet + '/'
                print('loadpath[202]:', loadpath)
                save_path = tf.train.latest_checkpoint(loadpath)
                if save_path is not None:
                    loader.restore(sess, save_path)

            elif ((epoch_idx / len(dataNames)) != 0):
                if args.pretrained != 0:
                    intOuts = dict()
                    intOuts[m_train] = list()
                    intOuts[m_dev] = list()
                    intOuts[m_test] = list()
                    for d_sub in modelDict:
                        if d_sub == dataSet:
                            continue
                        else:
                            loadpath = './modelSave/' + expName + '/' + d_sub + '/'
                            print('loadpath[224]:', loadpath)
                            save_path = tf.train.latest_checkpoint(loadpath)
                            if save_path is not None:
                                loader.restore(sess, save_path)
                            intOuts[m_train].append(
                                modelDict[d_sub]['runner'].info1epoch(m_train, modelDict[dataSet]['runner'], sess))
                            intOuts[m_dev].append(
                                modelDict[d_sub]['runner'].info1epoch(m_dev, modelDict[dataSet]['runner'], sess))
                            intOuts[m_test].append(
                                modelDict[d_sub]['runner'].info1epoch(m_test, modelDict[dataSet]['runner'], sess))

                loadpath = './modelSave/' + expName + '/' + dataSet + '/'
                print('loadpath[225]:', loadpath)
                save_path = tf.train.latest_checkpoint(loadpath)
                if save_path is not None:
                    loader.restore(sess, save_path)

            print(f" Start training {dataSet} model epoch {epoch_idx + 1} / {args.epoch}")
            (l, sl, tra, trsPara, train_duration) = modelDict[dataSet]['runner'].train1epoch(
                sess, batch_idx, infoInput=intOuts, tbWriter=tbWriter)

            print("== Epoch:%4d == | time : %d Min | train epoch duration: %d s |"
                  "\n train loss: %.6f" % (
            epoch_idx + 1, (time.time() - startTime) / 60, train_duration, l))
            modelDict[dataSet]['trainLossList'].append(l)
            modelDict[dataSet]['trainDurationList'].append(train_duration)

            (t_predictionResult, t_prfValResult, t_prfValWOCRFResult,
             test_x, test_ans, test_len, test_loss, test_duration) = modelDict[dataSet]['runner'].dev1epoch(m_test, trsPara, sess,
                                                                                  infoInput=intOuts, epoch=epoch_idx)

            # prfValResult: (precision, recall, f1)

            print("== Epoch:%4d == | time : %d Min | test epoch duration: %d s |"
                  "\n test loss: %.6f" % (
            epoch_idx + 1, (time.time() - startTime) / 60, test_duration, test_loss))
            modelDict[dataSet]['testLossList'].append(test_loss)
            modelDict[dataSet]['testDurationList'].append(test_duration)

            modelDict[dataSet]['precisionValList'].append(t_prfValResult[0])
            modelDict[dataSet]['recallValList'].append(t_prfValResult[1])
            modelDict[dataSet]['f1ValList'].append(t_prfValResult[2])
            saver.save(sess, './modelSave/' + expName + '/' + m_name + '/modelSaved')
            pickle.dump(trsPara, open('./modelSave/' + expName + '/' + m_name + '/trs_param.pickle', 'wb'))

            # if ((epoch_idx / len(dataNames)) == early_stops[epoch_idx % len(dataNames)]):
            modelDict[dataSet]['early_stop'] = True
            modelDict[dataSet]['maxF1'] = t_prfValResult[2]
            modelDict[dataSet]['stop_counter'] = 0
            modelDict[dataSet]['maxF1idx'] = epoch_idx
            modelDict[dataSet]['trs_param'] = trsPara
            modelDict[dataSet]['maxF1_x'] = test_x[:]
            modelDict[dataSet]['maxF1_ans'] = test_ans[:]
            modelDict[dataSet]['maxF1_len'] = test_len[:]
            pickle.dump(modelDict[dataSet]['maxF1idx'],
                        open('./modelSave/' + expName + '/' + dataSet + '/maxF1idx.pickle', 'wb'))
            if args.pretrained != 0:
                pickle.dump(intOuts[m_test],
                            open('./modelSave/' + expName + '/' + dataSet + '/bestInouts.pickle', 'wb'))

            for didx, dname in enumerate(dataNames):
                if not modelDict[dname]['early_stop']:
                    esFlag = False
                    break
                if modelDict[dname]['early_stop'] and didx == len(dataNames) - 1:
                    esFlag = True

            # create a metrics dict with only pickle-able values
            metricsDict = defaultdict(dict)
            metricsDict[dataSet]['trainLossList'] = modelDict[dataSet]['trainLossList']
            metricsDict[dataSet]['testLossList'] = modelDict[dataSet]['testLossList']
            metricsDict[dataSet]['trainDurationList'] = modelDict[dataSet]['trainDurationList']
            metricsDict[dataSet]['testDurationList'] = modelDict[dataSet]['testDurationList']
            metricsDict[dataSet]['f1ValList'] = modelDict[dataSet]['f1ValList']
            metricsDict[dataSet]['precisionValList'] = modelDict[dataSet]['precisionValList']
            metricsDict[dataSet]['recallValList'] = modelDict[dataSet]['recallValList']
            metricsDict[dataSet]['maxF1'] = modelDict[dataSet]['maxF1']
            metricsDict[dataSet]['maxF1idx'] = modelDict[dataSet]['maxF1idx']
            metricsDict[dataSet]['prevF1'] = modelDict[dataSet]['prevF1']
            metricsDict[dataSet]['stop_counter'] = modelDict[dataSet]['stop_counter']
            metricsDict[dataSet]['early_stop'] = modelDict[dataSet]['early_stop']
            metricsDict[dataSet]['m_name'] = modelDict[dataSet]['m_name']

            # save modelDict to the file after each epoch, overwrite the previous one
            metrics_path = './modelSave/' + expName + '/' + dataSet + '/metricsDict.pickle'
            pickle.dump(metricsDict, open(metrics_path, 'wb'))

            if esFlag:
                break

    # Get test result for each model
    for dataSet in modelDict:
        m_name = modelDict[dataSet]['args'].guidee_data
        print('====' + m_name.upper() + "_MODEL Test=====")
        with tf.compat.v1.Session(config=gpu_config) as sess:
            random.seed(seedV)
            np.random.seed(seedV)
            tf.compat.v1.set_random_seed(seedV)
            sess.run(tf.compat.v1.global_variables_initializer())
            loader = tf.compat.v1.train.Saver(tf.compat.v1.trainable_variables(), max_to_keep=10000)
            loadpath = './modelSave/' + expName + '/' + m_name + '/'

            if args.pretrained != 0:
                intOuts = dict()
                intOuts[m_test] = pickle.load(open(loadpath + 'bestInouts.pickle', 'rb'))
            else:
                intOuts = None

            trsPara = pickle.load(open(loadpath + 'trs_param.pickle', 'rb'))
            print('loadpath[224]:', loadpath)
            save_path = tf.train.latest_checkpoint(loadpath)
            if save_path is not None:
                loader.restore(sess, save_path)

            if modelDict[dataSet]['args'].tensorboard:
                tbWriter = tf.compat.v1.summary.FileWriter('test')
            else:
                tbWriter = None

            (t_predictionResult, t_prfValResult, t_prfValWOCRFResult,
             test_x, test_ans, test_len, test_loss, test_duration) = modelDict[dataSet]['runner'].dev1epoch(m_test, trsPara, sess,
                                                                                  infoInput=intOuts, epoch=None,
                                                                                  report=True)
