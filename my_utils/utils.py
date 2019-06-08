import re

def token_extend(reg_rules):
    return ' ' + reg_rules.group(0) + ' '

def reform_text(text):
    ## re.sub用于替换字符串中的匹配项
    text = re.sub(u'-|¢|¥|€|£|\u2010|\u2011|\u2012|\u2013|\u2014|\u2015|%|\[|\]|:|\(|\)|/', token_extend, text)
    text = text.strip(' \n')
    text = re.sub('\s+', ' ', text)
    return text
