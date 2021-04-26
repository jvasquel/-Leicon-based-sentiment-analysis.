 
import os
import re
import math
import string
import requests
import json
from itertools import product
from inspect import getsourcefile
from io import open

B_INCR = 0.293
B_DECR = -0.293
C_INCR = 0.733
N_SCALAR = -1.0

# for removing punctuation
REGEX_REMOVE_PUNCTUATION = re.compile('[%s]' % re.escape(string.punctuation))

PUNC_LIST = [".", "!", "?", ",", ";", ":", "-", "'", "\"","¡¡","¡¡¡",
             "!!", "!!!","¿¿", "??","¿¿¿", "???","¿¡¿", "?!?"]
NEGATE = \
    [ "nop",
         "tampoco", "nones",  "jamas",
     "nunca",  
     "sin", "no" , "raramente", "desprecio"]

BOOSTER_DICT = \
    {"absolutamente": B_INCR, "increiblemente": B_INCR, "muy": B_INCR, "mucho": B_INCR, "muchisimo": B_INCR, "demasiado": B_INCR, "completamente": B_INCR, "considerablemete": B_INCR,
     "decididamente": B_INCR, "profundamente": B_INCR, "condenado": B_INCR, "enormemente": B_INCR,
     "enteramente": B_INCR, "especialmente": B_INCR, "exepcionalmente": B_INCR, "extremadamente": B_INCR,
     "fabulosamente": B_INCR, "voltear": B_INCR, "flippin": B_INCR,
     "maldito": B_INCR, "maldito": B_INCR, "maldito": B_INCR, "puto": B_INCR, "completamente": B_INCR, "jodido": B_INCR,
     "grandiosamente": B_INCR, "super": B_INCR, "altamente": B_INCR, "sumamente": B_INCR, "increiblemente": B_INCR,
     "intensely": B_INCR, "majorly": B_INCR, "more": B_INCR, "most": B_INCR, "particularly": B_INCR,
     "puramente": B_INCR, "bastante": B_INCR, "realmente": B_INCR, "remarcable": B_INCR,
     "so": B_INCR, "substancialmente": B_INCR,
     "a fondo": B_INCR, "totamente": B_INCR, "tremendamente": B_INCR,
     "tremendo": B_INCR, "increiblemente": B_INCR, "inusualmente": B_INCR, "absolutamente": B_INCR,
     "muy": B_INCR,
     "casi": B_DECR, "escasamente": B_DECR, "dificilmente": B_DECR, "suficientemente": B_DECR,
     "a tope": B_DECR,  "kind-of": B_DECR,
     "menos": B_DECR, "pequeño": B_DECR, "marginalmente": B_DECR, "ocasionalmente": B_DECR, "parcialmente": B_DECR,
     "escasamente": B_DECR, "ligeramente": B_DECR, "en tanto": B_DECR,
     "suerte de ": B_DECR, "sorta": B_DECR,  }
# #Static methods# #

def negated(input_words, include_nt=True):
    input_words = [str(w).lower() for w in input_words]
    neg_words = []
    neg_words.extend(NEGATE)
    
    for word in neg_words:
        if word in input_words:
            return True
    if "no" in input_words:
        return True
    return False


def normalize(score, alpha=15):
    norm_score = score / math.sqrt((score * score) + alpha)
    if norm_score < -1.0:
        return -1.0
    elif norm_score > 1.0:
        return 1.0
    else:
        return norm_score


def allcap_differential(words):
    is_different = False
    allcap_words = 0
    for word in words:
        if word.isupper():
            allcap_words += 1
    cap_differential = len(words) - allcap_words
    if 0 < cap_differential < len(words):
        is_different = True
    return is_different


def scalar_inc_dec(word, valence, is_cap_diff):
    scalar = 0.0
    word_lower = word.lower()
    if word_lower in BOOSTER_DICT:
        scalar = BOOSTER_DICT[word_lower]
        if valence < 0:
            scalar *= -1
        # check if booster/dampener word is in ALLCAPS (while others aren't)
        if word.isupper() and is_cap_diff:
            if valence > 0:
                scalar += C_INCR
            else:
                scalar -= C_INCR
    return scalar


class SentiText(object):
    def __init__(self, text):
        if not isinstance(text, str):
            text = str(text).encode('utf-8')
        self.text = text
        self.words_and_emoticons = self._words_and_emoticons()
        # doesn't separate words from\
        # adjacent punctuation (keeps emoticons & contractions)
        self.is_cap_diff = allcap_differential(self.words_and_emoticons)

    def _words_plus_punc(self):
        no_punc_text = REGEX_REMOVE_PUNCTUATION.sub('', self.text)
        # removes punctuation (but loses emoticons & contractions)
        words_only = no_punc_text.split()
        # remove singletons
        words_only = set(w for w in words_only if len(w) > 1)
        # the product gives ('cat', ',') and (',', 'cat')
        punc_before = {''.join(p): p[1] for p in product(PUNC_LIST, words_only)}
        punc_after = {''.join(p): p[0] for p in product(words_only, PUNC_LIST)}
        words_punc_dict = punc_before
        words_punc_dict.update(punc_after)
        return words_punc_dict

    def _words_and_emoticons(self):
        wes = self.text.split()
        words_punc_dict = self._words_plus_punc()
        wes = [we for we in wes if len(we) > 1]
        for i, we in enumerate(wes):
            if we in words_punc_dict:
                wes[i] = words_punc_dict[we]
        return wes


class SentimentIntensityAnalyzer(object):
    def __init__(self, lexicon_file="vader_prueba.txt", emoji_lexicon="emoji_utf8_lexicon.txt",sadness="sadness.txt",joy="joy.txt",anger="anger.txt",fear="fear.txt",disgust="disgust.txt",surprise="surprise.txt"):
        _this_module_file_path_ = os.path.abspath(getsourcefile(lambda: 0))
        lexicon_full_filepath = os.path.join(os.path.dirname(_this_module_file_path_), lexicon_file)
        with open(lexicon_full_filepath, encoding='utf-8') as f:
            self.lexicon_full_filepath = f.read()
        self.lexicon = self.make_lex_dict()
        print(" __init__\n")


        _this_module_file_path_ = os.path.abspath(getsourcefile(lambda: 0))
        sadness_full_filepath = os.path.join(os.path.dirname(_this_module_file_path_), sadness)
        with open(sadness_full_filepath, encoding='utf-8') as f1:
            self.sadness_full_filepath = f1.read()
        self.sadness = self.make_sadness_dict()

        _this_module_file_path_ = os.path.abspath(getsourcefile(lambda: 0))
        joy_full_filepath = os.path.join(os.path.dirname(_this_module_file_path_), joy)
        with open(joy_full_filepath, encoding='utf-8') as f2:
            self.joy_full_filepath = f2.read()
        self.joy = self.make_joy_dict()

        _this_module_file_path_ = os.path.abspath(getsourcefile(lambda: 0))
        anger_full_filepath = os.path.join(os.path.dirname(_this_module_file_path_), anger)
        with open(anger_full_filepath, encoding='utf-8') as f3:
            self.anger_full_filepath = f3.read()
        self.anger = self.make_anger_dict()

        _this_module_file_path_ = os.path.abspath(getsourcefile(lambda: 0))
        fear_full_filepath = os.path.join(os.path.dirname(_this_module_file_path_), fear)
        with open(fear_full_filepath, encoding='utf-8') as f4:
            self.fear_full_filepath = f4.read()
        self.fear = self.make_fear_dict()

        _this_module_file_path_ = os.path.abspath(getsourcefile(lambda: 0))
        disgust_full_filepath = os.path.join(os.path.dirname(_this_module_file_path_), disgust)
        with open(disgust_full_filepath, encoding='utf-8') as f5:
            self.disgust_full_filepath = f5.read()
        self.disgust = self.make_disgust_dict()

        _this_module_file_path_ = os.path.abspath(getsourcefile(lambda: 0))
        surprise_full_filepath = os.path.join(os.path.dirname(_this_module_file_path_), surprise)
        with open(surprise_full_filepath, encoding='utf-8') as f6:
            self.surprise_full_filepath = f6.read()
        self.surprise = self.make_surprise_dict()


    def make_lex_dict(self):
        cont= 1;
        lex_dict = {}
        for line in self.lexicon_full_filepath.split('\n'): 

            (word, measure) = line.strip().split('\t')[0:2]

           # print(word ,"--",measure)
            if measure =="positivo":
                lex_dict[word] = float(1)
            if measure =="negativo":
                lex_dict[word] = float(-1)
            if measure =="neutro":
                lex_dict[word] = float(0)
        return lex_dict

    def make_sadness_dict(self):
        sadness_dict = {}
        for line in self.sadness_full_filepath.split('\n'): 

            (word) = line.strip().split('\t')[0]
            sadness_dict[word] = "sadness"
        return sadness_dict

    def make_joy_dict(self):
        joy_dict = {}
        for line in self.joy_full_filepath.split('\n'): 
            (word) = line.strip().split('\t')[0]
            joy_dict[word] = "joy"
        return joy_dict

    def make_anger_dict(self):
        anger_dict = {}
        for line in self.anger_full_filepath.split('\n'): 

            (word) = line.strip().split('\t')[0]
            anger_dict[word] = "anger"
        return anger_dict

    def make_fear_dict(self):
        fear_dict = {}
        for line in self.fear_full_filepath.split('\n'): 

            (word) = line.strip().split('\t')[0]
            fear_dict[word] = "fear"
        return fear_dict
    def make_disgust_dict(self):
        disgust_dict = {}
        for line in self.disgust_full_filepath.split('\n'): 

            (word) = line.strip().split('\t')[0]
            disgust_dict[word] = "fear"
        return disgust_dict

    def make_surprise_dict(self):
        surprise_dict = {}
        for line in self.surprise_full_filepath.split('\n'): 

            (word) = line.strip().split('\t')[0]
            surprise_dict[word] = "fear"
        return surprise_dict

    def polarity_scores(self, text):
        text_token_list = text.split()
        text_no_emoji_lst = []
        for token in text_token_list:
           
            text_no_emoji_lst.append(token)
        text = " ".join(x for x in text_no_emoji_lst)

        sentitext = SentiText(text)

        sentiments = []
        afects=[0,0,0,0,0,0]
        words_and_emoticons = sentitext.words_and_emoticons
        for item in words_and_emoticons:
            item_lowercase = item.lower()
            if item_lowercase in self.sadness:
                afects[0]=afects[0]+1
            if item_lowercase in self.joy:
                afects[1]=afects[1]+1
            if item_lowercase in self.anger:
                afects[2]=afects[2]+1
            if item_lowercase in self.fear:
                afects[3]=afects[3]+1
            if item_lowercase in self.disgust:
                afects[4]=afects[4]+1
            if item_lowercase in self.surprise:
                afects[5]=afects[5]+1

            valence = 0
            i = words_and_emoticons.index(item)
            if item.lower() in BOOSTER_DICT:
                sentiments.append(valence)
                continue
            if (i < len(words_and_emoticons) - 1 and item.lower() == "kind" and
                    words_and_emoticons[i + 1].lower() == "of"):
                sentiments.append(valence)
                continue

            sentiments = self.sentiment_valence(valence, sentitext, item, i, sentiments)
 

        sentiments = self._but_check(words_and_emoticons, sentiments)
        valence_dict = self.score_valence(sentiments, text,afects)
     
        return valence_dict

    def sentiment_valence(self, valence, sentitext, item, i, sentiments):
        is_cap_diff = sentitext.is_cap_diff
        words_and_emoticons = sentitext.words_and_emoticons
        item_lowercase = item.lower()
        if item_lowercase in self.lexicon:
            valence = self.lexicon[item_lowercase]
            if item.isupper() and is_cap_diff:
                if valence > 0:
                    valence += C_INCR
                else:
                    valence -= C_INCR

            for start_i in range(0, 3):
                if i > start_i and words_and_emoticons[i - (start_i + 1)].lower() not in self.lexicon:
                    s = scalar_inc_dec(words_and_emoticons[i - (start_i + 1)], valence, is_cap_diff)
                    if start_i == 1 and s != 0:
                        s = s * 0.9
                    if start_i == 2 and s != 0:
                        s = s * 0.9
                    valence = valence + s
                    valence = self._negation_check(valence, words_and_emoticons, start_i, i)
                   
                        

            valence = self._least_check(valence, words_and_emoticons, i)
        sentiments.append(valence)
        return sentiments

    def _least_check(self, valence, words_and_emoticons, i):
        if i > 1 and words_and_emoticons[i - 1].lower() not in self.lexicon \
                and words_and_emoticons[i - 1].lower() == "poco":
            if words_and_emoticons[i - 2].lower() != "en" and words_and_emoticons[i - 2].lower() != "mucho":
                valence = valence * N_SCALAR
        elif i > 0 and words_and_emoticons[i - 1].lower() not in self.lexicon \
                and words_and_emoticons[i - 1].lower() == "poco":
            valence = valence * N_SCALAR
        return valence

    @staticmethod
    def _but_check(words_and_emoticons, sentiments):
        words_and_emoticons_lower = [str(w).lower() for w in words_and_emoticons]
        if 'pero' in words_and_emoticons_lower:
            bi = words_and_emoticons_lower.index('pero')
            for sentiment in sentiments:
                si = sentiments.index(sentiment)
                if si < bi:
                    sentiments.pop(si)
                    sentiments.insert(si, sentiment * 0.5)
                elif si > bi:
                    sentiments.pop(si)
                    sentiments.insert(si, sentiment * 1.5)
        return sentiments
 

    @staticmethod
    def _negation_check(valence, words_and_emoticons, start_i, i):
        words_and_emoticons_lower = [str(w).lower() for w in words_and_emoticons]
        if start_i == 0:

            if negated([words_and_emoticons_lower[i - (start_i + 2)]]):
                valence = valence * N_SCALAR
        if start_i == 1:
            if words_and_emoticons_lower[i - 2] == "nunca" and \
                    (words_and_emoticons_lower[i - 1] == "no" or
                     words_and_emoticons_lower[i - 1] == "sin"):
                valence = valence * 1.25
            elif words_and_emoticons_lower[i - 2] == "sin" and \
                    words_and_emoticons_lower[i - 1] == "sin":
                valence = valence
            elif negated([words_and_emoticons_lower[i - (start_i + 1)]]): 
                valence = valence * N_SCALAR
        if start_i == 2:
            if words_and_emoticons_lower[i - 3] == "nunca" and \
                    (words_and_emoticons_lower[i - 2] == "so" or words_and_emoticons_lower[i - 2] == "este") or \
                    (words_and_emoticons_lower[i - 1] == "so" or words_and_emoticons_lower[i - 1] == "esto"):
                valence = valence * 1.25
            elif words_and_emoticons_lower[i - 3] == "sin" and \
                    (words_and_emoticons_lower[i - 2] == "sin" or words_and_emoticons_lower[i - 1] == "sin"):
                valence = valence
            elif negated([words_and_emoticons_lower[i - (start_i + 1)]]): 
                valence = valence * N_SCALAR
        return valence

    def _punctuation_emphasis(self, text):
        ep_amplifier = self._amplify_ep(text)
        qm_amplifier = self._amplify_qm(text)
        punct_emph_amplifier = ep_amplifier + qm_amplifier
        return punct_emph_amplifier

    @staticmethod
    def _amplify_ep(text):
        ep_count = text.count("!")
        if ep_count > 4:
            ep_count = 4
        ep_amplifier = ep_count * 0.292
        return ep_amplifier

    @staticmethod
    def _amplify_qm(text):
        qm_count = text.count("?")
        qm_amplifier = 0
        if qm_count > 1:
            if qm_count <= 3:
                qm_amplifier = qm_count * 0.18
            else:
                qm_amplifier = 0.96
        return qm_amplifier

    @staticmethod
    def _sift_sentiment_scores(sentiments):
        pos_sum = 0.0
        neg_sum = 0.0
        neu_count = 0
        for sentiment_score in sentiments:
            if sentiment_score > 0:
                pos_sum += (float(sentiment_score) + 1)
            if sentiment_score < 0:
                neg_sum += (float(sentiment_score) - 1)  
            if sentiment_score == 0:
                neu_count += 1
        return pos_sum, neg_sum, neu_count

    def score_valence(self, sentiments, text, afects):
        if sentiments:
            sum_s = float(sum(sentiments))
            sum_f = float(sum(afects))
            punct_emph_amplifier = self._punctuation_emphasis(text)
            if sum_s > 0:
                sum_s += punct_emph_amplifier
            elif sum_s < 0:
                sum_s -= punct_emph_amplifier

            compound = normalize(sum_s)
            pos_sum, neg_sum, neu_count = self._sift_sentiment_scores(sentiments)

            if pos_sum > math.fabs(neg_sum):
                pos_sum += punct_emph_amplifier
            elif pos_sum < math.fabs(neg_sum):
                neg_sum -= punct_emph_amplifier

            total = pos_sum + math.fabs(neg_sum) + neu_count

            pos = math.fabs(pos_sum / total)
            neg = math.fabs(neg_sum / total)
            neu = math.fabs(neu_count / total)
           
        else:
            compound = 0.0
            pos = 0.0
            neg = 0.0
            neu = 0.0
        if sum_f>0:
            sadness=math.fabs(afects[0] / sum_f)
            joy=math.fabs(afects[1] / sum_f)
            anger=math.fabs(afects[2] / sum_f)
            fear=math.fabs(afects[3] / sum_f)
            disgust=math.fabs(afects[4] / sum_f)
            surprise=math.fabs(afects[5] / sum_f)
        else:
            sadness=0.0
            joy=0.0
            fear=0.0
            disgust=0.0
            surprise=0.0
            anger=0.0

        if pos ==neg:
            polarity="neutro"

        if pos>neg:
            polarity="positivo"
        if neg>pos:
            polarity="negativo"


        sentiment_dict = \
            {"polarity":polarity,
             "anger": round(anger, 3),
             "disgust": round(disgust, 3),
             "fear": round(fear, 3),
             "joy": round(joy, 3),
             "sadness": round(sadness, 3),
             "surprise": round(surprise, 3)}

        return sentiment_dict


if __name__ == '__main__':

    analyzer = SentimentIntensityAnalyzer()
    print("----------------------------------------------------")
    senten= input("insert an sentence and press Enter: \n")
    print (senten,"\n")
  
    vsn = analyzer.polarity_scores(senten)
    print("{:-<65} {}".format(senten, str(vsn))) 

    print("----------------------------------------------------")
  
