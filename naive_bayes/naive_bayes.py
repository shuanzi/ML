# -*- coding: utf-8 -*-

from utils.vocab_utils import text_parse

__author__ = 'Xiquan Dai'

from sqlite3 import dbapi2 as sqlite, OperationalError


class Basement:
    def setdb(self, dbfile):
        self.con = sqlite.connect(dbfile)
        self.con.execute(
            'create table if not exists feature_topic_count(feature, topic, count)')
        self.con.execute('create table if not exists topic_count(topic, count)')

    def insert_feature_by_topic(self, feature, topic):
        tryCount = 0
        while tryCount < 5:
            try:
                res = self.con.execute(
                    'select count from feature_topic_count where feature="%s" and topic="%s"' % (
                        feature, topic)).fetchone()
                # 如果没有找到这条记录则插入
                if res == None:
                    self.con.execute(
                        'insert into feature_topic_count (feature, topic, count) values ("%s", "%s", 1)' % (
                            feature, topic))
                else:
                    count = float(res[0])
                    self.con.execute(
                        'update feature_topic_count set count=%d where feature="%s" and topic="%s"' % (
                            count + 1, feature, topic))
                return
            except OperationalError as error:
                tryCount += 1
                print(error)
                continue

    def insert_topic_count(self, topic):
        res = self.con.execute('select count from topic_count where topic="%s"' % topic).fetchone()
        if res is None:
            self.con.execute('insert into topic_count (topic, count) values ("%s", 1)' % topic)
        else:
            count = float(res[0])
            self.con.execute(
                'update topic_count set count=%d where topic="%s"' % (count + 1, topic))

    def feature_count(self, f, topic):
        res = self.con.execute(
            'select count from feature_topic_count where feature="%s" and topic="%s"' % (
                f, topic)).fetchone()
        if res is None:
            return 0
        else:
            return float(res[0])

    def topic_count(self, topic):
        res = self.con.execute('select count from topic_count where topic="%s"' % topic).fetchone()
        if res is None:
            return 0
        else:
            return float(res[0])

    def total_count(self):
        res = self.con.execute('select count from topic_count').fetchall()
        return sum(res[i][0] for i in range(len(res)))

    def all_topics(self):
        res = self.con.execute('select topic from topic_count').fetchall()
        return [res[i][0] for i in range(len(res))]

    def close_db(self):
        self.con.close()


class NaiveBayesClassifier(Basement):
    def __init__(self):
        Basement.__init__(self)
        self.thresholds = {}

    # 初始化提取特征的方法函数
    def get_features(self, text):
        if text:
            return text_parse(text)
        return []

    # 计算概率，计算P(f|topic)条件概率，即特征f在类别topic条件下出现的概率
    def feature_under_topic_prob(self, f, topic):
        # 如果该类别文档数为0则返回0
        if self.topic_count(topic) == 0:
            return 0
        return float(self.feature_count(f, topic)) / float(self.topic_count(topic))

    # 对fprob的条件概率计算方法进行优化，设置初始概率为0.5，权值为1
    # 参数： f为特征, topic为类别，weight为初始值ap所占权重
    def weighted_prob(self, f, topic, weight=1.0, ap=0.5):
        # 计算当前条件概率
        basic_prob = self.feature_under_topic_prob(f, topic)
        # 统计特征在所有分类中出现的次数
        totals = sum([self.feature_count(f, c) for c in self.all_topics()])
        # 计算加权平均
        bp = ((weight * ap) + (totals * basic_prob)) / (weight + totals)
        return bp

    # 计算整片文章的属于topic类的概率  等于文章中所有单词属于topic类的条件概率之积
    def docprob(self, item, topic):
        features = self.get_features(item)
        # 将所有特征的概率相乘
        p = 1
        for f in features:
            p *= self.weighted_prob(f, topic)
        return p

    # P(topic|item) = P(item|topic) * P(topic) / P(item)
    # 其中 P(item) 这一项由于不参与比较，因此可以忽略
    def prob(self, item, topic):
        topicprob = float(self.topic_count(topic)) / float(self.total_count())
        docprob = self.docprob(item, topic) * topicprob
        return docprob

    # 暂定阈值0.8
    def getthresholds(self):
        return 0.8

    def train(self, item, topic):
        features = self.get_features(item)
        # 根据分类添增加特征计数值
        for f in features:
            # print("training: do insert feature: %s, topic: %s" % (f, topic))
            self.insert_feature_by_topic(f, topic)
        # 增加分类计数值
        self.insert_topic_count(topic)
        self.con.commit()
        print("training done %s size %s" % (topic, features.__len__()))

    # 根据prob计算出的所有topic类的P(topic|item)进行比较，同时根据设定的thresholds阈值，将item判定到某一类
    def classify(self, item, default=None):
        # 构建所有类别的概率并排序
        prob = sorted([(topic, self.prob(item, topic)) for topic in self.all_topics()],
                      key=lambda x: x[1], reverse=True)

        # 如果 最大概率 > 阈值 * 次大概率，则判断为最大概率所属类别，否则判定为default类别
        if prob[0][1] > self.getthresholds() * prob[1][1]:
            return prob[0][0]
        else:
            return default
