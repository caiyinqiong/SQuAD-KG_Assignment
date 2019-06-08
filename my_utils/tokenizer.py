
class Vocabulary(object):
    INIT_LEN = 4
    def __init__(self):
        self.tok2ind = {'PADPAD': 0, 'UNKUNK': 1, 'BOSBOS': 2, 'EOSEOS': 3}
        self.ind2tok = {0: 'PADPAD', 1: 'UNKUNK', 2: 'BOSBOS', 3: 'EOSEOS'}

    def __len__(self):
        return len(self.tok2ind)

    def __iter__(self):
        return iter(self.tok2ind)

    def __contains__(self, key):
        if type(key) == int:
            return key in self.ind2tok
        elif type(key) == str:
            return key in self.tok2ind
    
    def __getitem__(self, key):
        if type(key) == int:
            return self.ind2tok.get(key, 'UNKUNK')
        if type(key) == str:
            return self.tok2ind.get(key, self.tok2ind.get('UNKUNK'))
    
    def __setitem__(self, key, item):
        if type(key) == int and type(item) == str:
            self.ind2tok[key] = item
        elif type(key) == str and type(item) == int:
            self.tok2ind[key] = item

    def add(self, token):
        if token not in self.tok2ind:
            index = len(self.tok2ind)
            self.tok2ind[token] = index
            self.ind2tok[index] = token

    def get_vocab_list(self):
        return [k for k in self.tok2ind.keys()]

    def toidx(self, tokens):
        return [self[token] for token in tokens]

    def copy(self):
        new_vocab = Vocabulary()
        for w in self:
            new_vocab.add(w)
        return new_vocab
    
    def build(words):
        vocab = Vocabulary()
        for word in words:
            vocab.add(word)
        return vocab
