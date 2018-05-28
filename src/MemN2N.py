from keras.engine.topology import Layer
from keras import backend as K


class MemN2N(Layer):

    def __init__(self, max_story_size, sentence_size, vocab_size, embedding_size, k_hop, **kwargs):
        self.max_story_size = max_story_size
        self.sentence_size = sentence_size
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.k_hop = k_hop
        super(MemN2N, self).__init__(**kwargs)

    def build(self, input_shape):
        self.emb_A = [0] * (self.k_hop + 1)
        for i in range(self.k_hop + 1):
            self.emb_A[i] = self.add_weight(name='Embedding_' + str(i),
                                shape=(self.vocab_size, self.embedding_size),
                                initializer='uniform',
                                trainable = True)

        super(MemN2N, self).build(input_shape)
        
    def call(self, inputs):
        story = inputs[0]
        question = inputs[1]
        
        if K.dtype(story) != 'int32':
            story = K.cast(story, 'int32')
        if K.dtype(question) != 'int32':
            question = K.cast(question, 'int32')
            
        u_emb = K.sum(K.gather(self.emb_A[0], question), axis=1)
        for i in range(self.k_hop) :
            m_emb = K.sum(K.gather(self.emb_A[i], story), axis=2)
            c_emb = K.sum(K.gather(self.emb_A[i + 1], story), axis=2)
            
            u_temp = K.tile(K.expand_dims(u_emb, 1), (1, self.max_story_size, 1))
            probs = K.softmax(K.sum(m_emb * u_temp, axis=2))
            probs = K.tile(K.expand_dims(probs, 2), (1, 1, self.embedding_size))
            u_emb = K.sum(c_emb * probs, axis=1) + u_emb
 
        output = K.dot(u_emb,  K.transpose(self.emb_A[-1]))
        output = K.softmax(output)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.vocab_size)
    
