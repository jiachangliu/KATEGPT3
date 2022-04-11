import os

categories = dict()

categories["SST-2"] = dict()
categories["SST-2"]["Qs"] = ["Sentence"]
categories["SST-2"]["A"] = ["Label"]

categories["IMDB"] = dict()
categories["IMDB"]["Qs"] = ["Sentence"]
categories["IMDB"]["A"] = ["Label"]

categories["IMDB_UnitTest300"] = dict()
categories["IMDB_UnitTest300"]["Qs"] = ["Sentence"]
categories["IMDB_UnitTest300"]["A"] = ["Label"]

categories["CoLA"] = dict()
categories["CoLA"]["Qs"] = ["Sentence"]
categories["CoLA"]["A"] = ["Label"]

categories["QNLI"] = dict()
categories["QNLI"]["Qs"] = ["Sentence"]
categories["QNLI"]["A"] = ["Label"]

categories["QQP"] = dict()
categories["QQP"]["Qs"] = ["Question1", "Question2"]
categories["QQP"]["A"] = ["Is_duplicate"]

categories["RTE"] = dict()
categories["RTE"]["Qs"] = ["Sentence1", "Sentence2"]
categories["RTE"]["A"] = ["Label"]

categories["STS-B"] = dict()
categories["STS-B"]["Qs"] = ["Sentence1", "Sentence2"]
categories["STS-B"]["A"] = ["Score"]

categories["MNLI"] = dict()
categories["MNLI"]["Qs"] = ["Sentence1", "Sentence2"]
categories["MNLI"]["A"] = ["Label1"]

categories["WNLI"] = dict()
categories["WNLI"]["Qs"] = ["Sentence1", "Sentence2"]
categories["WNLI"]["A"] = ["Label"]

categories["MRPC"] = dict()
categories["MRPC"]["Qs"] = ["#1 String", "#2 String"]
categories["MRPC"]["A"] = ["Quality"]

# categories["trivia_qa_UnitTest_train_100000_dev_200"] = dict()
# categories["trivia_qa_UnitTest_train_100000_dev_200"]["Qs"] = ["Q"]
# categories["trivia_qa_UnitTest_train_100000_dev_200"]["A"] = ["A"]
# 
# categories["trivia_qa_UnitTest_train_100000_dev_1000"] = dict()
# categories["trivia_qa_UnitTest_train_100000_dev_1000"]["Qs"] = ["Q"]
# categories["trivia_qa_UnitTest_train_100000_dev_1000"]["A"] = ["A"]
# 
# categories["trivia_qa_train_78785_dev_full"] = dict()
# categories["trivia_qa_train_78785_dev_full"]["Qs"] = ["Q"]
# categories["trivia_qa_train_78785_dev_full"]["A"] = ["A"]
# 
# categories["web_qs_train_3778_dev_full"] = dict()
# categories["web_qs_train_3778_dev_full"]["Qs"] = ["Q"]
# categories["web_qs_train_3778_dev_full"]["A"] = ["A"]
# 
# categories["web_qs_train_3417_dev_full"] = dict()
# categories["web_qs_train_3417_dev_full"]["Qs"] = ["Q"]
# categories["web_qs_train_3417_dev_full"]["A"] = ["A"]
# 
# categories["natural_qs_train_79168_dev_full"] = dict()
# categories["natural_qs_train_79168_dev_full"]["Qs"] = ["Q"]
# categories["natural_qs_train_79168_dev_full"]["A"] = ["A"]

categories["ToTTo"] = dict()
categories["ToTTo"]["Qs"] = ["Table"]
categories["ToTTo"]["A"] = ["Sentence"]

categories["MT"] = dict()
categories["MT"]["Qs"] = ["Source"]
categories["MT"]["A"] = ["Target"]

categories["sentiment"] = dict()
categories["sentiment"]["Qs"] = ["Sentence"]
categories["sentiment"]["A"] = ["Label"]

categories["text2code"] = dict()
categories["text2code"]["Qs"] = ["Text"]
categories["text2code"]["A"] = ["Code"]

categories["QA"] = dict()
categories["QA"]["Qs"] = ["Q"]
categories["QA"]["A"] = ["A"]

templates = dict()

templates["MT"] = dict()
templates["MT"]["Qs"] = ['{} ==']
templates["MT"]["A"] = [' {}\n\n', '']

templates["sentiment"] = dict()
templates["sentiment"]["Qs"] = ['Sentence:{} ']
templates["sentiment"]["A"] = ['Label:{}\n\n', 'Label:']

templates["ToTTo"] = dict()
templates["ToTTo"]["Qs"] = ['Table:{} ']
templates["ToTTo"]["A"] = ['Sentence:{}\n\n', 'Sentence:']

templates["text2code"] = dict()
templates["text2code"]["Qs"] = ["Text:{} "]
templates["text2code"]["A"] = ["Code:{}\n\n", 'Code:']

templates["QA"] = dict()
templates["QA"]["Qs"] = ["Q: {}\n"]
templates["QA"]["A"] = ["A: {}\n\n", "A:"]

# -*- coding: utf-8 -*-
# https://stackoverflow.com/questions/4576077/how-can-i-split-a-text-into-sentences
import re
alphabets= "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr|mr|st|mrs|ms|dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov)"

digits = "([0-9])"

def split_into_sentences(text):
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)

    text = re.sub(digits + "[.]" + digits,"\\1<prd>\\2",text)

    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    return sentences

# https://stackoverflow.com/questions/1518522/find-the-most-common-element-in-a-list
from collections import Counter
def most_common(lst):
    data = Counter(lst)
    return max(lst, key=data.get)

# https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
#     for i in range(0, len(lst), n):
#         yield lst[i:i + n]
    return [lst[i:i + n] for i in range(0, len(lst), n)]

def truncSent(sentence, words_count=175):
    new_sent = sentence.split(' ')
    if len(new_sent) > words_count:
        new_sent = ' '.join(new_sent[:words_count])
        return new_sent
    return sentence

def constructPrompt(df, labels, indices, templates, Q_list, A_list, A_flag=False, truncate=False):
    tmp_example = ''
    Q_templates = templates['Qs']
    A_templates = templates['A']

    for tmp_index in indices:
        for i, Q in enumerate(Q_list):
            if truncate:
                tmp_example += Q_templates[i].format(truncSent(df.loc[tmp_index, Q.lower()]))
            else:
                tmp_example += Q_templates[i].format(df.loc[tmp_index, Q.lower()])
        if A_flag:
            for A in A_list:
                if truncate:
                    tmp_example += A_templates[0].format(truncSent(df.loc[tmp_index, A.lower()]))
                else:
                    tmp_example += A_templates[0].format(df.loc[tmp_index, A.lower()])
        else:
            for A in A_list:
                tmp_example += A_templates[1]
    return tmp_example


#    tmp_example = ""
#    for tmp_index in indices:
#        for Q in Q_list:
#            if truncate:
#                tmp_example += Q + ': ' + truncSent(df.loc[tmp_index, Q.lower()]) + '\n'
#            else:
#                tmp_example += Q + ': ' + df.loc[tmp_index, Q.lower()] + '\n'
#        if A_flag:
#            for A in A_list:
#                if truncate:
#                    tmp_example += A + ': ' + truncSent(df.loc[tmp_index, A.lower()]) + '\n'
#                else:
#                    tmp_example += A + ': ' + df.loc[tmp_index, A.lower()] + '\n'
##                 tmp_example += A + ':' + labels[tmp_index] + '\n'
#            tmp_example += '\n'
#        else:
#            for A in A_list:
#                tmp_example += A + ':'
#    return tmp_example

def cleanLabel(labels, digits):
    for i, label in enumerate(labels):
        if label == 1:
            if digits:
                labels[i] = '1'
            else:
                labels[i] = 'positive'
        elif label == 0:
            if digits:
                labels[i] = '0'
            else:
                labels[i] = 'negative'

        if label == "entailment":
            labels[i] = 'entailment'
        elif label == "not_entailment":
            labels[i] = 'non-entailment'
    return labels