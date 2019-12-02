# 根据生成word_index(包含train+test数据集), 在glove中找到对应的embedding vec，保存为npz文件
import numpy as np
from data_helper.data_utils import *


def load_glove_embedding(word_index, file=None, trimmed_filename=None, load=False, dim=300):
    if load == True:  #
        with np.load(trimmed_filename) as data:
            return data["embeddings"]
    else:
        embeddings_index = {}
        with open(file, encoding='utf8') as f:
            for line in f:
                values = line.rstrip().split(" ")
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs

        print('Preparing embedding matrix.')
        embedding_matrix = np.zeros((len(word_index) + 1, dim))
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
        np.savez_compressed(trimmed_filename, embeddings=embedding_matrix)  #

    print("Embedding load done！")
    return embedding_matrix


if __name__ == '__main__':

    def hehe(train_path, test_path, save_path=None):
        print(train_path)
        print(test_path)
        train_sens, _, _ = load_sens_labels_ml(train_path)
        test_sens, _, _ = load_sens_labels_ml(test_path)
        widx = get_widx(train_sens + test_sens)
        embedding_matrix = load_glove_embedding(
            word_index=widx,
            file=r'D:\NLP程序相关\数据集收集\glove词向量\glove.840B.300d.txt',
            trimmed_filename=save_path,
            load=True
        )
        return embedding_matrix

    # lap14:
    # hehe(lap14_path[lap14_train_path], lap14_path[lap14_test_path], r'D:\NLP程序相关\MyModel2\data\laptop14')

    # res14:
    # hehe(rest14_path[rest14_train_path], rest14_path[rest14_test_path], r'D:\NLP程序相关\MyModel2\data\rest14')

    # rest15
    # hehe(rest15_path[rest15_train_path], rest15_path[rest15_test_path], r'D:\NLP程序相关\MyModel2\data\rest15')

    # rest16
    # hehe(rest16_path[rest16_train_path], rest16_path[rest16_test_path], r'D:\NLP程序相关\MyModel2\data\rest16')


