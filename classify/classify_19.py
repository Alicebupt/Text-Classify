# Created by Helic on 2018/6/20
# 数字有选择的替换
from gensim import corpora, models
from scipy.sparse import csr_matrix
import numpy as np
import os, re, time, logging
import pandas
import jieba
import pickle as pkl
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
# from sklearn.grid_search import GridSearchCV 
from sklearn.model_selection import GridSearchCV
import copy

def read_data_from_csv(csv_path):
    """read data from train.csv"""
    df = pandas.read_csv(csv_path)
    df = df[['label', 'texts']].fillna(method='pad')  # 前一个值填充后面的NAN
    df.sample(frac=1).reset_index(drop=True)
    return df

def convert_text_to_wordlist(text, cut_all_flag=False):
    """将文本转化为分词后的列表
    input:
        text: string
        cut_all_flag:如果cut_all=False，则会列出最优的分割选项；如果cut_all=True, 则会列出所有可能出现的词
    output:
        word_list: ['a', 'b']"""
    word_list = remove_stopwords(jieba.cut(text.strip(), cut_all=cut_all_flag))  # 分词
    return word_list

def get_stopwords(stopwords_path):
    """stopwords.txt文件中，每行放一个停用词，以\n分隔"""
    stopwords = []
    with open(stopwords_path, 'r', encoding='utf-8') as f:
        for item in f.readlines():
            stopwords.append(item.strip())
    stopwords.extend(['，', '。', '？', '“', '”', '‘', '’', '；', '：', '！', '、', '（', '）', '-', '=',
                      '【', '】', ' ', '{', '}', ',', '.', '/', '\\', '(', ')', '?', '!', ';', ':', '\'',
                      '"', '[', ']', '~', '\n', '\t'])
    return set(stopwords)

def remove_stopwords(words):
    """去掉一些停用词和数字"""
    stopwords = get_stopwords(stopwords_path="data/stopwords.txt")
    new_words = []
    for i in words:
        if i in stopwords or i.isdigit():  # 去除停用词和数字
            continue
        else:
            new_words.append(i)
    return new_words

def safe_mkdir(path):
    """ Create a directory if there isn't one already. """
    try:
        os.mkdir(path)
    except OSError:  # 当目录已经存在时会抛出异常
        pass

def svm_classify(train_set, train_tag, test_set, test_tag):
    # clf = svm.LinearSVC()
    # clf = svm.SVC(kernel="linear", C=2, probability=True)
    # grid = GridSearchCV(svm.SVC(kernel="rbf",probability=True), param_grid={"C":[0.1, 1, 10], "gamma": [1, 0.1, 0.01]}, cv=4)
    # grid.fit(train_set, train_tag)
    # print("The best parameters are %s with a score of %0.2f"
    #   % (grid.best_params_, grid.best_score_))
    clf = svm.SVC(kernel="rbf", gamma=2, probability=True)
    clf_res = clf.fit(train_set, train_tag)
    print('训练完毕')

    train_pred = clf_res.predict(train_set)
    # test_pred = clf_res.predict(test_set)
    train_err_num, train_err_ratio = checkPred(train_tag, train_pred)
    #test_err_num, test_err_ratio = checkPred(test_tag, test_pred)
    print('=== SVM分类训练完毕，分类结果如下 ===')
    print('训练集准确率: {e}'.format(e=train_err_ratio))
    # print('测试集准确率: {e}'.format(e=test_err_ratio))

    train_pred = clf_res.predict_proba(train_set)
    #test_pred = clf_res.predict_proba(test_set)
    train_err_num, train_err_ratio = top_5(train_tag, train_pred)
    #test_err_num, test_err_ratio = top_5(test_tag, test_pred)

    print('=== SVM分类训练完毕，前top2分类结果如下 ===')
    print('训练集准确率: {e}'.format(e=train_err_ratio))
    #print('测试集准确率: {e}'.format(e=test_err_ratio))

    return clf_res

def checkPred(data_tag, data_pred):
    if data_tag.__len__() != data_pred.__len__():
        raise RuntimeError('The length of data tag and data pred should be the same')
    err_count = 0
    for i in range(data_tag.__len__()):
        if data_tag[i] != data_pred[i]:
            err_count += 1
    err_ratio = err_count / data_tag.__len__()
    return [err_count, 1-err_ratio]

def f(x):
    return catg_list[int(x)]

def top_5(data_tag, data_pred):
    if data_tag.__len__() != data_pred.__len__():
        raise RuntimeError('The length of data tag and data pred should be the same')
    err_count = 0
    for i in range(data_tag.__len__()):
        # 取前五
        top_5_list = []
        data_cache = copy.deepcopy(data_pred[i])
        data_pred[i].sort()
        for j in range(2):
            l = data_pred[i].__len__()-j-1
            top_5_list.append(np.where(data_cache==data_pred[i][l])[0])
        # print(data_tag[i])
        # print(top_5_list)
        if data_tag[i] not in top_5_list:
            err_count += 1
        # with open('data/result.txt','a',encoding='utf8') as f:
        #     f.write(catg_list[data_tag[i]]+'\n')
        #     top_5_list = map(f, top_5_list)
        #     f.write(str(top_5_list)+'\n')
    acc = 1-(err_count/data_tag.__len__())
    return err_count, acc

def test(model, test_path='data/test_19.csv'):
    with open(test_path, 'r', encoding='utf8') as f:
        test_data = f.readlines()
    count = 0
    err = 0
    for i in range(1,len(test_data)):
        label = test_data[i].strip().split(',')[0]
        text = test_data[i].strip().split(',')[1]
        label_pred = model.predict(text)
        count += 1
        if label_pred != label:
            err += 1
    print('all text: ', count)
    print('err : ', err)

class FirstClassifier:
    def __init__(self, catg_list, csv_path, dic_path="data/train.dict", tfidf_path="data/tfidf_corpus", lsi_path="data/lsi_corpus",
                    lsimodel_path="model/lsi_model.pkl", predictor_path="model/svm.pkl"):
        self.dictionary_path = dic_path
        self.path_tmp_tfidf = tfidf_path
        self.path_tmp_lsi = lsi_path
        self.dataframe = read_data_from_csv(csv_path)
        self.catg_list = catg_list
        self.catg_index = {}
        self.catg_backward_index = {}
        for i in range(len(self.catg_list)):
            self.catg_index[self.catg_list[i]] = i
            self.catg_backward_index[i] = self.catg_list[i]
        print(len(self.catg_list))
        print(self.catg_index)
        print(self.catg_backward_index)
        self.path_tmp_lsimodel = lsimodel_path
        self.path_tmp_predictor = predictor_path

    def generate_dictionary(self):
        """第一阶段:遍历文档，生成词典,并去掉频率较少的项;如果指定的位置没有词典，则重新生成一个。如果有，则跳过该阶段"""
        if self.dictionary_path and os.path.exists(self.dictionary_path):
            print("=== 检测到词典已经存在，跳过该阶段 ===")
            self.dictionary = corpora.Dictionary.load(self.dictionary_path)  # 如果跳过了第一阶段，则从指定位置读取词典
        else:
            print("=== 未检测到有词典存在，开始遍历生成词典 ===")
            self.dictionary = corpora.Dictionary()
            for text in self.dataframe['texts']:
                self.dictionary.add_documents([convert_text_to_wordlist(text, cut_all_flag=False)])
            # 去掉词典中出现次数过少的
            small_freq_ids = [tokenid for tokenid, docfreq in self.dictionary.dfs.items() if docfreq < 5]
            self.dictionary.filter_tokens(small_freq_ids)
            self.dictionary.compactify()  # 使得ID连续
            self.dictionary.save(self.dictionary_path)
            print('=== 词典已经生成 ===')

    def convert_text_to_tfidf(self):
        """第二阶段，将文档转化成tfidf向量"""
        if self.path_tmp_tfidf and os.path.exists(self.path_tmp_tfidf):
            print('=== 检测到tfidf向量已经生成，跳过该阶段 ===')
            # 从磁盘中读取corpus
            corpus_tfidf = {}
            for catg in self.catg_list:
                path = '{f}{s}{c}.mm'.format(f=self.path_tmp_tfidf, s=os.sep, c=self.catg_index[catg])# s=os.sep
                corpus = corpora.MmCorpus(path)
                corpus_tfidf[catg] = corpus
            print('=== tfidf文档读取完毕，开始转化成lsi向量 ===')
        else:
            print('=== 未检测到有tfidf文件夹存在，开始生成tfidf向量 ===')
            safe_mkdir(self.path_tmp_tfidf)
            print(self.path_tmp_tfidf)
            tfidf_model = models.TfidfModel(dictionary=self.dictionary)
            corpus_tfidf = {}
            for index, row in self.dataframe.iterrows():
                word_list = convert_text_to_wordlist(row['texts'], cut_all_flag=False)
                file_bow = self.dictionary.doc2bow(word_list)  # [(),()]在gensim中，向量是稀疏表示的。例如[(0,5),(6,3)],该向量的第0个
                # 元素值为5，第6个元素值为3，其他为0.
                # print(file_bow, word_list, row['texts'])
                file_tfidf = tfidf_model[file_bow]  # [(0, 0.3220636177678036), (1, 0.9467180288292361)]
                tmp = corpus_tfidf.get(row['label'], [])  # Python 字典(Dictionary) get() 函数返回指定键的值，如果值不在字典中返回默认值。
                tmp.append(file_tfidf)
                if len(tmp) == 1:
                    corpus_tfidf[row['label']] = tmp
            # 将tfidf中间结果储存起来
            catgs = list(corpus_tfidf.keys())
            for catg in catgs:
                corpora.MmCorpus.serialize('{f}{s}{c}.mm'.format(f=self.path_tmp_tfidf, s=os.sep, c=self.catg_index[catg]),
                                           corpus_tfidf.get(catg), id2word=self.dictionary)
                print('catg {c} has been transformed into tfidf vector'.format(c=catg))
            print('=== tfidf向量已经生成 ===')
        return corpus_tfidf

    def convert_tfidf_to_lsi(self, corpus_tfidf):
        """第三阶段，开始将tfidf转化成lsi"""
        if self.path_tmp_lsi and os.path.exists(self.path_tmp_lsi):
            print('=== 检测到lsi向量已经生成，跳过该阶段 ===')
            # 从磁盘中读取corpus
            corpus_lsi = {}
            for catg in self.catg_list:
                path = '{f}{s}{c}.mm'.format(f=self.path_tmp_lsi, s=os.sep, c=self.catg_index[catg])
                corpus = corpora.MmCorpus(path)
                corpus_lsi[catg] = corpus
            print('=== lsi向量读取完毕，开始进行分类 ===')
        else:
            print('=== 未检测到有lsi文件夹存在，开始生成lsi向量 ===')
            corpus_tfidf_total = []
            for catg in self.catg_list:
                tmp = corpus_tfidf.get(catg)
                corpus_tfidf_total += tmp
            lsi_model = models.LsiModel(corpus=corpus_tfidf_total, id2word=self.dictionary, num_topics=50)  # 50代表向量维度
            # 将lsi模型存储到磁盘上
            self.path_tmp_lsimodel = os.path.join('model')
            safe_mkdir(self.path_tmp_lsimodel)  # 只是创建一个目录
            self.path_tmp_lsimodel = os.path.join('lsi_model.pkl')
            with open(self.path_tmp_lsimodel, 'wb') as lsi_file:
                pkl.dump(lsi_model, lsi_file)
            del corpus_tfidf_total  # lsi model已经生成，释放变量空间
            print('=== lsi模型已经生成 ===')

            # 生成corpus of lsi, 并逐步释放corpus_tfidf变量空间
            corpus_lsi = {}
            # 保存lsi向量
            self.path_tmp_lsi = os.path.join('data', 'lsi_corpus')
            safe_mkdir(self.path_tmp_lsi)
            for catg in self.catg_list:
                corpu = [lsi_model[doc] for doc in corpus_tfidf.get(catg)]
                corpus_lsi[catg] = corpu
                corpus_tfidf.pop(catg)
                corpora.MmCorpus.serialize('{f}{s}{c}.mm'.format(f=self.path_tmp_lsi, s=os.sep, c=self.catg_index[catg]),
                                           corpu, id2word=self.dictionary)
            print('=== lsi向量已经生成 ===')
        return corpus_lsi

    def train(self, corpus_lsi, model_name="svm"):
        """train the classifier"""
        if 1 == 2: # self.path_tmp_predictor and os.path.exists(self.path_tmp_predictor):
            print('=== 检测到{}分类器已经生成，跳过该阶段 ==='.format(model_name))
        else:
            print('=== 未检测到{}分类器，开始训练 ==='.format(model_name))
            corpus_lsi_total = []
            tag_list = []
            doc_num_list = []
            for count, catg in enumerate(self.catg_list):
                tmp = corpus_lsi[catg]
                tag_list += [count] * tmp.__len__()
                doc_num_list.append(tmp.__len__())
                corpus_lsi_total += tmp
                corpus_lsi.pop(catg)
            # 将gensim中的mm表示转化成numpy矩阵表示
            data, rows, cols = [], [], []
            line_count = 0
            for line in corpus_lsi_total:
                for elem in line:
                    rows.append(line_count)
                    cols.append(elem[0])
                    data.append(elem[1])
                line_count += 1
            lsi_matrix = csr_matrix((data, (rows, cols))).toarray()
            # 生成训练集和测试集
            rarray = np.random.random(size=line_count)
            train_set, train_tag, test_set, test_tag = [], [], [], []
            for i in range(line_count):
                if rarray[i] < 0.9:
                    train_set.append(lsi_matrix[i, :])
                    train_tag.append(tag_list[i])
                else:
                    test_set.append(lsi_matrix[i, :])
                    test_tag.append(tag_list[i])
            print('train data : ', len(train_set))
            print('train tag : ', len(train_tag))
            print('test data : ', len(test_set))
            print('test data : ', len(test_tag))
            # 生成分类器
            if model_name == "svm":
                predictor = svm_classify(train_set, train_tag, test_set, test_tag)
            # elif model_name == "gbdt":
            #     predictor = gbdt_classify(train_set, train_tag, test_set, test_tag)
            # elif model_name == "lr":
            #     predictor = lr_classify(train_set, train_tag, test_set, test_tag)
            # elif model_name == "rf":
            #     predictor =  rf_classifier(train_set, train_tag, test_set, test_tag)
            # elif model_name == "nb":
            #     predictor = nb_classifier(train_set, train_tag, test_set, test_tag)
            # elif model_name == "knn":
            #     predictor = knn_classifier(train_set, train_tag, test_set, test_tag)
            # elif model_name == 'dt':
            #     predictor = dt_classifier(train_set, train_tag, test_set, test_tag)
            safe_mkdir("model")
            self.path_tmp_predictor = os.path.join("model", '{}.pkl'.format(model_name))
            with open(self.path_tmp_predictor, 'wb') as x:
                pkl.dump(predictor, x)
            print('=== 训练完成，{}分类器保存至{}==='.format(model_name, self.path_tmp_predictor))

    def predict(self, text):
        """对新文本进行判断, string"""
        dictionary = corpora.Dictionary.load(self.dictionary_path)
        with open('lsi_model.pkl', 'rb') as f:
            lsi_model = pkl.load(f)
        with open(self.path_tmp_predictor, 'rb') as f:
            predictor = pkl.load(f)
        word_list = convert_text_to_wordlist(text, cut_all_flag=False)
        file_bow = dictionary.doc2bow(word_list)  # [(),()]在gensim中，向量是稀疏表示的。例如[(0,5),(6,3)],该向量的第0个
        tfidf_model = models.TfidfModel(dictionary=dictionary)
        demo_tfidf = tfidf_model[file_bow]
        demo_lsi = lsi_model[demo_tfidf]
        data, cols, rows = [], [], []
        for item in demo_lsi:
            data.append(item[1])
            cols.append(item[0])
            rows.append(0)
        demo_matrix = csr_matrix((data, (rows, cols))).toarray()
        result = predictor.predict_proba(demo_matrix)
        result = result[0].tolist()
        result_index = result.index(max(result))
        return self.catg_list[result_index] # result[0]




if __name__ == '__main__':
    catg_list = ['微产品', '额度调整', '调额婉拒', '卡片挂失', '交易模式', '解除标志', '卡转卡办理', '卡片激活', '年费产品', 
    '设置密码', '现金分期', '延期还款', '溢缴款办理', '圆梦金大额消费分期', '账单单笔分期', '中收产品', '自动转账及购汇', 
    '查询修改账单日','分期提前缴款']
    # catg_list = ['保险业务', '额度办理', '分期业务', '高端增值服务', '还款服务', '积分服务', '交易查询', '卡片管理', '卡片邮寄', 
    #             '开卡设密', '其他业务', '渠道咨询', '申请审批', '优惠活动', '逾期业务', '账单查询', '账户服务', '中间业务', '资料修改/查询']
    Classifier_1 = FirstClassifier(catg_list, csv_path="data/train_19.csv", dic_path="data/train.dict", tfidf_path="data/tfidf_corpus", lsi_path="data/lsi_corpus",
                    lsimodel_path="model/lsi_model.pkl", predictor_path="model/svm.pkl")
    Classifier_1.generate_dictionary()
    corpus_tfidf = Classifier_1.convert_text_to_tfidf()
    corpus_lsi = Classifier_1.convert_tfidf_to_lsi(corpus_tfidf)

    Classifier_1.train(corpus_lsi, model_name='svm')
    # Classifier_1.train(corpus_lsi, model_name='gbdt')
    # Classifier_1.train(corpus_lsi, model_name='lr')
    # Classifier_1.train(corpus_lsi, model_name='rf')
    # Classifier_1.train(corpus_lsi, model_name='nb')
    # Classifier_1.train(corpus_lsi, model_name='knn')
    # Classifier_1.train(corpus_lsi, model_name='dt')

    test(model=Classifier_1)
