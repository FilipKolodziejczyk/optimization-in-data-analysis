import time

from ops.inputData import *
from ops.ops import *

import tensorflow as tf
tf.compat.v1.disable_eager_execution()

import tensorflow_addons as tfa


class RunModel:
    def __init__(self, model, args, ID2wordVecIdx, ID2char, expName, m_name, m_train='train', m_dev='dev', m_test='test'):
        self.model=model
        self.tbWriter=None
        
        self.args=args
        self.ID2wordVecIdx=ID2wordVecIdx
        self.ID2char=ID2char
        self.expName=expName
        self.m_name=m_name
        self.m_train=m_train
        self.m_dev=m_dev
        self.m_test=m_test

        self.batch_size = args.batch_size
        self.lr = args.lr
        self.lr_decay_counter = 0
        self.stop_counter = 0
        self.m_x_data, self.m_x_char_data, self.m_answerData, self.m_lengthData = input_datapickle(args.guidee_data, ID2wordVecIdx)
        self.m_batchgroup = batch_sort(self.m_lengthData, self.batch_size)

        self.overwrite_model_optimizer()

    def overwrite_model_optimizer(self):
        if self.args.optimizer == 'adam':
            self.model.train = tf.compat.v1.train.AdamOptimizer(learning_rate=self.lr).minimize(self.model.loss)
        elif self.args.optimizer == 'rmsprop':
            self.model.train = tf.compat.v1.train.RMSPropOptimizer(learning_rate=self.lr).minimize(self.model.loss)
        elif self.args.optimizer == 'adagrad':
            self.model.train = tf.compat.v1.train.AdagradOptimizer(learning_rate=self.lr).minimize(self.model.loss)
        else:
            raise ValueError("Invalid optimizer")
            

    def train1epoch(self, sess, batch_idx, infoInput=None, tbWriter=None): #infoInput not implimented yet
        start_time = time.perf_counter()
        for b_idx in batch_idx:
            x_minibatch, y_minibatch, xlen_minibatch, x_char_minibatch = idx2data(
                self.m_batchgroup[self.m_train][b_idx], self.m_x_data[self.m_train], self.m_x_char_data[self.m_train], 
                self.m_answerData[self.m_train], self.m_lengthData[self.m_train])
            x_minibatch, y_minibatch, x_char_minibatch, maxLen = batch_padding(
                x_minibatch, y_minibatch, xlen_minibatch, x_char_minibatch)
            x_charPad,x_charLen = char_padding(inputs=x_char_minibatch,
                        voca_size=len(self.ID2char), embedding_dim=self.args.ce_dim,
                        wordMaxLen=maxLen, charMaxLen=self.args.char_maxlen)
            
            infos = np.zeros([len(x_minibatch),maxLen,self.args.hidden_size*2])
            if infoInput is None:
                infoOuts = list()
                for i in range(5):
                    infoOuts.append(np.zeros([len(x_minibatch),maxLen,self.args.hidden_size*2])) 
            else:
                infoOuts = list()
                for infotmp in infoInput[self.m_train]:
                    infoOuts.append(infotmp[b_idx])
                tmpLen = len(infoOuts)
                for i in range(tmpLen,5):
                    infoOuts.append(np.zeros([len(x_minibatch), maxLen, self.args.hidden_size*2]))
             
            feed_dict1 = {self.model.X: x_minibatch,
                          self.model.Y: y_minibatch, 
                          self.model.X_len: xlen_minibatch,
                          self.model.X_char: x_charPad, 
                          self.model.X_char_len: x_charLen,
                          self.model.maxLen: maxLen, 
                          self.model.lr: self.lr,
                          self.model.infos: infos,
                          self.model.infos1: infoOuts[0],
                          self.model.infos2: infoOuts[1],
                          self.model.infos3: infoOuts[2],
                          self.model.infos4: infoOuts[3],
                          self.model.infos5: infoOuts[4],
                          self.model.emb_dropout: self.args.embdropout,
                          self.model.lstm_dropout: self.args.lstmdropout
                          }

            if b_idx==0 and self.args.tensorboard:
                summary, l, sl, tra, trsPara = sess.run([self.model.summaryMerged,
                                            self.model.loss, 
                                            self.model.sequence_loss,
                                            self.model.train, 
                                            self.model.transition_params],
                                            feed_dict=feed_dict1)
                self.tbWriter.add_summary(summary, self.global_step)
            else:
                l, sl, tra, trsPara = sess.run([self.model.loss, 
                                                self.model.sequence_loss,
                                                self.model.train, 
                                                self.model.transition_params],
                                                feed_dict=feed_dict1)
        duration = time.perf_counter() - start_time
        return (l, sl, tra, trsPara, duration) # Note the indent. It only gives last batch logit of 1 epoch training

    def dev1epoch(self, data, trsPara, sess, infoInput=None, epoch=None, report=False): #data : dev or test?
        start_time = time.perf_counter()
        predictionResult=list()
        viterbi_scoreList=list()
        predictionWOCRFResult=list()
        dev_x = list()
        dev_ans = list()
        dev_len = list()
        total_loss = 0

        for b_idx in range(len(self.m_batchgroup[data])):
            x_minibatch, y_minibatch, xlen_minibatch, x_char_minibatch = idx2data(
                self.m_batchgroup[data][b_idx], self.m_x_data[data], self.m_x_char_data[data], 
                self.m_answerData[data], self.m_lengthData[data])
            x_minibatch, y_minibatch, x_char_minibatch, maxLen = batch_padding(
                x_minibatch, y_minibatch, xlen_minibatch, x_char_minibatch)
            x_charPad,x_charLen = char_padding(inputs=x_char_minibatch,
                        voca_size=len(self.ID2char), embedding_dim=self.args.ce_dim,
                        wordMaxLen=maxLen, charMaxLen=self.args.char_maxlen)
            dev_x.extend(x_minibatch)
            dev_ans.extend(y_minibatch)
            dev_len.extend(xlen_minibatch)
            
            infos = np.zeros([len(x_minibatch),maxLen,self.args.hidden_size*2])
            if infoInput is None:
                infoOuts = list()
                for i in range(5):
                    infoOuts.append(np.zeros([len(x_minibatch),maxLen,self.args.hidden_size*2])) 
            else:
                infoOuts = list()
                for infotmp in infoInput[data]:
                    infoOuts.append(infotmp[b_idx])
                tmpLen = len(infoOuts)
                for i in range(tmpLen,5):
                    infoOuts.append(np.zeros([len(x_minibatch),maxLen,self.args.hidden_size*2]))
            
            feed_dict2 = {self.model.X: x_minibatch, 
                          self.model.Y: y_minibatch, 
                          self.model.X_len: xlen_minibatch,
                          self.model.X_char: x_charPad, 
                          self.model.X_char_len: x_charLen,
                          self.model.maxLen: maxLen, 
                          self.model.lr: self.lr,
                          self.model.infos: infos, #don't use
                          self.model.infos1: infoOuts[0],
                          self.model.infos2: infoOuts[1],
                          self.model.infos3: infoOuts[2],
                          self.model.infos4: infoOuts[3],
                          self.model.infos5: infoOuts[4],
                          self.model.emb_dropout: 0,
                          self.model.lstm_dropout: 0
                          }
            
            logitsPridict, loss = sess.run([self.model.logits, self.model.loss], feed_dict=feed_dict2)
            total_loss += loss

            for sentence in logitsPridict:
                viterbi, viterbi_score=tfa.text.crf.viterbi_decode(sentence,trsPara)
                predictionResult.append(viterbi)
                viterbi_scoreList.append(viterbi_score)
            predictionWOCRFResult.extend(sess.run(self.model.prediction, feed_dict=feed_dict2))
        
        predictionResult = viterbi_pp(predictionResult, dev_len, self.args.num_class)
        prfValResult=prf(predictionResult,dev_ans,dev_len)

        validation_loss = total_loss / len(self.m_batchgroup[data])

        if data == self.m_dev:
            infoschk = sess.run([self.model.infos1_w, self.model.infos2_w, self.model.infos3_w, self.model.infos4_w, self.model.infos5_w])                                                                                   
            print("Learning Rate : %.4f"%(self.lr))
            self.lr = self.lr*(1-self.args.lr_decay)
            if int(epoch/6)==30 and self.args.lr_pump:
                self.lr = self.args.lr
        if report:
            print("[%s] Precision : %.4f | Recall : %.4f | F1: %.4f"%(data, prfValResult[0],prfValResult[1],prfValResult[2]))

        prfValWOCRFResult=None
        duration = time.perf_counter() - start_time
        
        return (predictionResult, prfValResult, prfValWOCRFResult, dev_x, dev_ans, dev_len, validation_loss, duration)

    def info1epoch(self, data, dataset, sess):
        # data : train, dev or test?
        # dataset : data set, define in modelDcit[dataSet]['runner']
        
        m_x_data = dataset.m_x_data
        m_x_char_data = dataset.m_x_char_data
        m_answerData = dataset.m_answerData
        m_lengthData = dataset.m_lengthData
        m_batchgroup = dataset.m_batchgroup
        lstmOuts = list() 
        for b_idx in range(len(m_batchgroup[data])):
            x_minibatch, y_minibatch, xlen_minibatch, x_char_minibatch = idx2data(
                m_batchgroup[data][b_idx], m_x_data[data], m_x_char_data[data], 
                m_answerData[data], m_lengthData[data])
            x_minibatch, y_minibatch, x_char_minibatch, maxLen = batch_padding(
                x_minibatch, y_minibatch, xlen_minibatch, x_char_minibatch)
            x_charPad,x_charLen = char_padding(inputs=x_char_minibatch,
                        voca_size=len(self.ID2char), embedding_dim=self.args.ce_dim,
                        wordMaxLen=maxLen, charMaxLen=self.args.char_maxlen)
            
            infoInputtmp = np.zeros([len(x_minibatch),maxLen,self.args.hidden_size*2])

            feed_dict2 = {self.model.X: x_minibatch, 
                          self.model.Y: y_minibatch, 
                          self.model.X_len: xlen_minibatch,
                          self.model.X_char: x_charPad, 
                          self.model.X_char_len: x_charLen,
                          self.model.maxLen: maxLen, 
                          self.model.lr: self.lr,
                          self.model.infos:infoInputtmp,
                          self.model.infos1: infoInputtmp,
                          self.model.infos2: infoInputtmp,
                          self.model.infos3: infoInputtmp,
                          self.model.infos4: infoInputtmp,
                          self.model.infos5: infoInputtmp,
                          self.model.emb_dropout: 0,
                          self.model.lstm_dropout: 0
                          }
            
            lstmOut = sess.run(self.model.outputs_concat, feed_dict=feed_dict2)
            lstmOuts.append(lstmOut)
        return lstmOuts 
