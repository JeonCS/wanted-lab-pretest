import re

from konlpy.tag import Okt

from constants import STOP_WORDS


class Preprocessor:
    def __init__(self):
        self.okt = Okt()
    # Preprocess Data
    def _clean_words(self, x):
        x = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', x) # remove link
        x = re.sub(r'[@%\\*=)~#&\+á?\xc3\xa1\-\|\.\:\;\!\-\,\_\~\$\'\"\•\]]', '',x) #remove punctuation
        # x = re.sub(r'[/(\[]', ' ',x) # add a space with opening bracket or slash
        # x = re.sub(r'\b\w\b', '',x)
        x = re.sub(r'\s+', ' ', x) #remove spaces 
        x = re.sub(r"^\s+", '', x) #remove space from start 
        x = re.sub(r'\s+$', '', x) #remove space from the end
        x = re.sub(r'\d+','', x)# remove number 
        x = x.lower() #lower case 
        return x

    def split_text(self, x):
        x = self._clean_words(x)
        x = self.okt.morphs(x)
        return [a for a in x if len(a) > 1 and a not in STOP_WORDS]
