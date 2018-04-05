import string
from nltk.corpus import stopwords
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim import corpora, models, similarities

def text_preprocess(text):
    # split into words
    from nltk.tokenize import word_tokenize
    tokens = word_tokenize(text)
    # convert to lower case
    tokens = [w.lower() for w in tokens]
    # remove punctuation from each word
    import string
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    words = [word for word in stripped if word.isalpha()]
    # filter out stop words
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]
    # stemming of words
    from nltk.stem.porter import PorterStemmer
    porter = PorterStemmer()
    stemmed = [porter.stem(word) for word in words]
    # delete words with length 1
    final = [word for word in stemmed if len(word) is not 1]
    return final


def texts_preprocess(input_dict):
    output_dict = {}
    texts = list(input_dict.values())
    for ind in range(len(texts)):
        texts[ind] = text_preprocess(texts[ind])
    from collections import defaultdict
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1
    output_dict = {key: [token for token in texts[ind] if frequency[token] > 1] for key, ind in zip(input_dict.keys(), range(len(texts)))}
    return output_dict


def LDA(texts, num_topics=15):
    # 根据文本生成字典
    dictionary = corpora.Dictionary(texts)
#     print(dictionary)
#     V = len(dictionary)
    
    # 根据字典，将每行文档都转换为索引的形式
    corpus = [dictionary.doc2bow(text) for text in texts]
    # 逐行打印
#     for line in corpus:
#         print(line)
        
    # print('LDA Model:')
    # 训练模型
    lda = models.LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary,
                          alpha='auto', eta='auto', minimum_probability=0.001)
    
    # 计算相似度
    index = similarities.MatrixSimilarity(lda[corpus])
    return index