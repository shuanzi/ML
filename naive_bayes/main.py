import random
from multiprocessing.dummy import Pool as ThreadPool

from utils.file_utils import get_all_dir, get_all_file_path, _open
from .naive_bayes import NaiveBayesClassifier


def do_classify(argv):
    text = argv[0]
    topic = argv[1]
    file_path = argv[2]
    cl2 = NaiveBayesClassifier()
    cl2.setdb('/Users/daixiquan/Documents/Keep_document/naive.sqlite3')
    res4 = cl2.classify(text)
    cl2.close_db()
    print("classify %s, topic:%s" % (file_path, res4))
    if res4 == topic:
        return 1
    else:
        return 0


# 一次训练多条数据，入参为 classifier 类的一个对象
def do_training(argv_list):
    text = argv_list[0]
    topic = argv_list[1]
    if text:
        cl = NaiveBayesClassifier()
        cl.setdb('/Users/daixiquan/Documents/Keep_document/naive.sqlite3')
        cl.train(text, topic)
        cl.close_db()


if __name__ == "__main__":
    test_root_dir = "test_subject"

    dirList = get_all_dir(test_root_dir)
    ALL_TOPICS = dirList

    # 读取数据开始训练
    training = []
    test = []
    for topic in ALL_TOPICS:
        file_path_list = get_all_file_path(test_root_dir + "/" + topic)
        for file_path in file_path_list:
            text = _open(file_path)
            arg = []
            arg.append(text)
            arg.append(topic)
            arg.append(file_path)
            if random.randrange(10) > 3:
                training.append(arg)
            else:
                test.append(arg)
    pool = ThreadPool(1)
    pool.map(do_training, training)
    pool.close()

    pool = ThreadPool(16)
    # 构建朴素贝叶斯分类器，直接使用已经建立好的数据库
    files = get_all_file_path(test_root_dir)
    result = pool.map(do_classify, test)
    pool.close()
