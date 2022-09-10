import contractions
import string

class Vocabulary:
    def __init__(self, dataset_df, key):
        sentence_list = self.preprocess(dataset_df, key)
        max_len = -1
        for sen in sentence_list:
            if len(sen) > max_len:
                max_len = len(sen)
        print("Max len: "+key, max_len)
        self.stoi, self.itos = Vocabulary.create_stoi_itos(sentence_list)
        self.vocab_size = len(self.stoi)
        print("Language: " + key + ", len: " + str(self.vocab_size))

    @staticmethod
    def create_stoi_itos(list_sentences):
        stoi = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}
        word_set = list(set(" ".join(list_sentences).split(" ")))
        for i, word in enumerate(word_set):
            stoi[word] = i + 4
        itos = {i: w for i, w in enumerate(stoi)}
        return stoi, itos

    def preprocess(self, dataset_df, key):
        # Expand contractions
        dataset_df[key] = dataset_df[key].apply(lambda x: contractions.fix(x))

        # Remove punctuation   
        dataset_df[key] = dataset_df[key].apply(lambda x: ''.join(ch.lower() for ch in x if ch not in string.punctuation))

        #Strip
        dataset_df[key] = dataset_df[key].apply(lambda x: x.strip())
        return list(dataset_df[key])
    
    def convert_sen2num(self, sentence):
        num = [self.stoi["<SOS>"]]
        word_list = sentence.split(" ")
        for word in word_list:
            word = word.lower()
            if word in self.stoi:
                num.append(self.stoi[word])
            else:
                num.append(self.stoi['<UNK>'])
        num.append(self.stoi["<EOS>"])
        return num
    
    def convert_num2sen(self, num):
        sen = [self.itos[index] for index in num]
        return "".join(word+" " for word in sen)