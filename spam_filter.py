# LING 539 Final Project
# Information Retrieval and Machine Learning: SPAM E-mail Classification Task
# By Meng Jia

from __future__ import division
import email
import re
import email.charset
from sklearn.metrics import confusion_matrix
from nltk.classify.util import accuracy
from nltk.classify.maxent import MaxentClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import timeit

start_time = timeit.default_timer()


gold_file = 'trec07p/partial/index'
trec_prefix = 'trec07p/data/inmail.'
trec_size = 3000


def read_gold(filename):
    f = open(filename)
    gold = []
    for line in f:
        new_line = line.lower().split()
        if new_line[0] == 'spam':
            gold.append(1)
        else:
            gold.append(2)

    return gold


def load_trec(prefix, data_size, strip_html=False):
    messages = []
    subjects = []

    for i in range(1, data_size + 1):
        filename = prefix + str(i)
        f = open(filename, 'r').read()
        f_data = unicode(f, errors='ignore')

        msg = email.message_from_string(f_data)  # use email module to read the body
        sub = msg['Subject']  # read email subject
        raw = ''
        for part in msg.walk():
            if part.get_content_type() == 'text/html' or part.get_content_type() == 'text/plain':
                raw += part.get_payload()
                raw = re.sub('\s+', ' ', raw)
                if strip_html is True:
                    raw = re.sub('<[^<]+?>', '', raw)
        messages.append(raw)
        subjects.append(sub)

    if strip_html is True:
        return messages
    else:
        return messages, subjects


def cosine_similarity_calculation(spam_ham, query):  # use sklearn to vectorize documents, tfidf weights applied
    train_vectorizer = TfidfVectorizer(stop_words='english', decode_error='replace')
    train_matrix = train_vectorizer.fit_transform(spam_ham)
    vocab = train_vectorizer.get_feature_names()

    query_vectorizer = TfidfVectorizer(stop_words='english', decode_error='replace', vocabulary=vocab)
    query_matrix = query_vectorizer.fit_transform([query])

    vector_avg = train_matrix.mean(axis=0)  # average the vectors of all training documents
    cos_sim = cosine_similarity(query_matrix, vector_avg)  # calculate similarity of averaged vectors and query vector

    return cos_sim[0][0]  # return a matrix which contains only one number, get that number


def feature_extractor(raw_datum, clean_datum, sub):
    features = {}
    features['htmltags'] = len(re.findall('<[^<]+?>', raw_datum))  # feature 1

    features['spam cosine'] = cosine_similarity_calculation(train_spams, clean_datum)  # feature 2
    features['ham cosine'] = cosine_similarity_calculation(train_hams, clean_datum)

    if sub is None:
        pass
    else:
        features['has(@)'] = sub.lower().count('@')  # feature 3

        if sub.startswith('=?'):  # feature 4
            features['weird Sub'] = True
        else:
            features['weird Sub'] = False

    spam_words = ['cialis', 'viagra', 'money', 'offer', 'pills', 'doctor', 'finance', 'discount',
                  'pharmacy', 'medication', 'stock', 'credit', 'mortgage', 'moneyback', '$', '@']
    for sw in spam_words:
        features['count({})'.format(sw)] = raw_datum.lower().count(sw)  # feature 5

    return features


def results(train, query_data, query_no_label, query_labels):
    print '\nCalculating final results...'
    megam_classifier = MaxentClassifier.train(train, 'megam')  # build and train the maxent classifier
    accu = accuracy(megam_classifier, query_data)  # calculate the classification accuracy

    predicted = megam_classifier.classify_many(query_no_label)  # get a list of predicted labels
    cm = confusion_matrix(query_labels, predicted)  # build confusion matrix

    return accu, cm


# -------------
# MAIN FUNCTION
# -------------

print '\nLoading gold labels...'
gold_labels = read_gold(gold_file)
train_label, dev_label, test_label = gold_labels[:1000], gold_labels[1000:2000], gold_labels[2000:trec_size]

print '\nLoading TREC data...'
raw_trec, subjects = load_trec(trec_prefix, trec_size, strip_html=False)
trec_no_html = load_trec(trec_prefix, trec_size, strip_html=True)


train_spams = [d for d, l in zip(trec_no_html, gold_labels) if l == 1]  # a list of spams in training corpus
train_hams = [d for d, l in zip(trec_no_html, gold_labels) if l == 2]  # hams in training


print '\nExtracting features...'
featuresets = [(feature_extractor(d1, d2, sub), l) for d1, d2, l, sub in
               zip(raw_trec, trec_no_html, gold_labels, subjects)]
train_data, dev_data, test_data = featuresets[:1000], featuresets[1000:2000], featuresets[2000:trec_size]


no_label_feature = [pair[0] for pair in featuresets]
train_no, dev_no, test_no = no_label_feature[:1000], no_label_feature[1000:2000], no_label_feature[2000:trec_size]


# the performance on development corpus
dev_accu, dev_cm = results(train_data, dev_data, dev_no, dev_label)
print '\nClassification accuracy on development corpus: %s' % dev_accu
print dev_cm


# the performance on testing corpus
test_accu, test_cm = results(train_data, test_data, test_no, test_label)
print '\nClassification accuracy on test corpus: %s' % test_accu
print test_cm


# the total running time of program
print '\nRunning Time: %s sec\n' % "{0:.2f}".format(timeit.default_timer() - start_time)
