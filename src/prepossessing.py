from .data_utils import *
from functools import reduce
from itertools import chain

def prepossessing(task_id, task_path):
    train, test = load_task(task_path, task_id)
    data = train + test

    vocab = sorted(reduce(lambda x, y: x | y, (set(list(chain.from_iterable(s)) + q + a) for s, q, a in data)))
    word_idx = dict((c, i + 1) for i, c in enumerate(vocab))

    max_story_size = max(map(len, (s for s, _, _ in data)))
    mean_story_size = int(np.mean([ len(s) for s, _, _ in data ]))
    sentence_size = max(map(len, chain.from_iterable(s for s, _, _ in data)))
    query_size = max(map(len, (q for _, q, _ in data)))
    memory_size = min(320, max_story_size)

    vocab_size = len(word_idx) + 1 # +1 for nil word
    sentence_size = max(query_size, sentence_size) # for the position

    trainS, trainQ, trainA = vectorize_data(train, word_idx, sentence_size, memory_size)
    testS, testQ, testA = vectorize_data(test, word_idx, sentence_size, memory_size)

    print("task_id", task_id)
    print("Vocab size", vocab_size)
    print("Longest sentence length", sentence_size)
    print("Longest story length", max_story_size)
    print("Average story length", mean_story_size)
    print("Query size", query_size)
    
    print('-')
#    print('trainS shape:', trainS.shape)
#    print('testS shape:', testS.shape)
#    print('-')
#    print('trainQ shape:', trainQ.shape)
#    print('testQ shape:', testQ.shape)
#    print('-')
#    print('trainA shape:', trainA.shape)
#    print('testA shape:', testA.shape)
#    print('-')
    
    return trainS, trainQ, trainA, testS, testQ, testA, max_story_size, sentence_size, vocab_size
