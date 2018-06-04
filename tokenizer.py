""" Detect end of sentences. """
from sklearn.svm import SVC

import pickle
import os
import time
import numpy as np


def timeit(method):
    """
    A decorator used for time profiling functions and methods
    :param method:
    :return: time in ms for a method
    """

    def timed(*args, **kwargs):
        timeStart = time.time()
        result = method(*args, **kwargs)
        timeEnd = time.time()

        if 'log_time' in kwargs:
            name = kwargs.get('log_name', method.__name__.upper())
            kwargs['log_time'][name] = int((timeEnd - timeStart) * 1000)
        else:
            print('%r %2.2f ms' % (method.__name__, (timeEnd - timeStart) * 1000))
        return result

    return timed


def ngram(text, n=2):
    """ Accepts string and returns ngram
    Args:
        text: the text of whose ngram is to be computed
        n: 1 for unigram, 2 for bigram and so on
    Returns:
        generator of ngrams
    """
    skip = n - 1
    # return [text[i:i + n] for i in range(len(text) - skip)][:]
    return (text[i:i + n] for i in range(len(text) - skip))


# Prepare dataset
@timeit
def prepareDataset(input_file='data/corpus.csv', partition=0.8):
    """
    Reads file.
    Partitions them into test and train data.
    """
    sent_delimiter = ["।", "|", "?", "॥"]
    text = open(file=input_file, mode='r', encoding='utf8').read()
    text = text.split()

    # Converting text into number
    vocab = list(set(text))
    char_to_id = {ch: id for id, ch in enumerate(vocab)}
    id_to_char = {id: ch for id, ch in enumerate(vocab)}

    # Objective is to predict if two words contain a sentence determiner between them.
    # This is the logic behind tri-grams.
    text = ngram(text, n=3)
    text = list(text)[:]  # convert generator to list

    # # todo: three list comprehensions over extremely large list do not make sense
    input = [item[::2] for item in text]
    input = [[char_to_id[a], char_to_id[b]] for [a, b] in input]
    # Check if the middle item in trigram is sentence delimiter
    labels = [1 if item[1] in sent_delimiter else 0 for item in text]

    # Fast method of obtaining inputs and labels. Not tested.
    # input, labels = [], []
    # for a,b,c in text:
    #     input.append([char_to_id[a], char_to_id[c]])
    #     labels.append(1 if b in sent_delimiter else 0)

    # To check if you generated correct input and labels
    print(input[:10])
    print(labels[:10])

    # Making the distribution more fair. Make len(0):len(1) almost equal.
    for i in range(4):
        for i in range(0, len(labels), 2):
            try:
                if labels[i] == 0:
                    del labels[i]
                    del input[i]
            except:
                break

    partition = int(len(input) * partition)
    train_input, train_labels, test_input, test_labels = input[:partition], labels[:partition], \
                                                         input[partition:], labels[partition:]
    return train_input, train_labels, test_input, test_labels, char_to_id, id_to_char


@timeit
def load_train_classifier(REPORT_ACCURACY=True, classifier_file="classifier.csv"):
    """
    Loads if the classifer already exists
    Else, trains a classifier and saves it
    Returns classifier, char_to_id, id_to_char
    """
    if not os.path.exists(classifier_file):
        print('Initializing fresh parameters')
        print("1. Prepare dataset")
        train_input, train_labels, test_input, test_labels, char_to_id, id_to_char = prepareDataset()
        print('No of train samples {}, test samples {}'.format(len(train_labels), len(test_input)))

        print("2. Initialize classifier and train")
        classifier = SVC()
        classifier.fit(X=train_input, y=train_labels)

        file = open(classifier_file, mode='wb')
        pickle.dump(classifier, file, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(char_to_id, file, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(id_to_char, file, protocol=pickle.HIGHEST_PROTOCOL)

        if REPORT_ACCURACY:
            print('Computing Accuracy on test set . . .')
            score = classifier.score(X=test_input, y=test_labels)
            print('Accuracy : ', "%.2f" % score)

    else:
        print('Reusing pretrained classifier')
        file = open(classifier_file, mode='rb')
        classifier = pickle.load(file)
        char_to_id = pickle.load(file)
        id_to_char = pickle.load(file)

    return classifier, char_to_id, id_to_char


class SentenceTokenizer:
    """ Tokenizes text without punctuation into sentences and returns list of sentences """

    def __init__(self):
        self.classifier, self.char_to_id, self.id_to_char = load_train_classifier(classifier_file='classifier.csv')

    def tokenize(self, text):
        """
        Algorithm:
            Generates bigrams.
            Classifier predicts if a bigram should have a sentence delimiter in between them.
                Add a delimiter when true
            Join the sentence.
            Return list of sentence(s)
        """

        # Check if user input is empty
        if text == '' or text is None:
            return text

        # Tokenize user text into words and compute bigrams
        text = text.split()
        bigrams = ngram(text)
        bigrams = [[self.char_to_id.get(a, -1), self.char_to_id.get(b, -1)] for [a, b] in bigrams]

        sentences = []
        for index, [word, bigram] in enumerate(zip(text[:len(bigrams)], bigrams)):
            bigram = np.int32(bigram).reshape((1, 2))  # convert bigram to numpy array
            delimiter = self.classifier.predict(bigram)  # predict if the bigram should contain a delimiter

            sentences.append(word)
            # If delimiter was found. Add a delimiter to the list of words in the sentence
            if delimiter:
                sentences.append(' || ')

        sentences.append(text[-1])

        # Join text and split them on sentence delimiter
        sentences = ' '.join(sentences)
        sentences = sentences.split(' || ')

        return sentences


if __name__ == '__main__':
    text = "ल्यु नेतृत्वको प्रतिनिधि मण्डल तीनदिने नेपाल भ्रमणका लागि आइतबार काठमाडौँ आएको हो नेपाल आएलगत्तै प्रतिनिधि मण्डलले राष्ट्रपति विद्यादेवी भण्डारीसँग शिष्टाचार भेट गरेको थियो"
    tokenizer = SentenceTokenizer()
    tokenizer.tokenize(text)
