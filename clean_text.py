from html.parser import HTMLParser
import string
import re

URL_REGEX = re.compile("http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+")
REPEATED_CHARACTER_REGEX = re.compile(r"(([A-z])\2{2,})")

class MLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.fed = []

    def handle_data(self, d):
        self.fed.append(d)

    def get_data(self):
        return ''.join(self.fed)


def remove_excess_whitespace_from_string(string):
    string = re.sub(' +', ' ',string).strip()
    return " ".join(string.split())


def remove_punctuation_from_string(s):
    return s.translate(None, string.punctuation)


def remove_html_from_string(string):
    stripper = MLStripper()
    stripper.feed(string)
    return stripper.get_data()

def replace_with_double_character(matchobj):
    return matchobj.group(2) * 2


def remove_repeated_alpha_chars(string):
    """
    Looks for runs of characters of 3 or more of the same thing, and then replaces it with just 2
    of that same character
    (useful for user text e.g. twitter, with strings like 'oooohhhhhhhh noooooooo' ->  'oohh noo'
    """
    return REPEATED_CHARACTER_REGEX.sub(replace_with_double_character, string)


def clean_string(string,
                 lowercase_characters=True,
                 remove_html=True,
                 remove_excess_whitespace=True,
                 replace_repeated_characters=True,
                 remove_urls=True
                 ):
    if remove_urls:
        string = URL_REGEX.sub('', string)
    if remove_html:
        string = remove_html_from_string(string)
    if replace_repeated_characters:
        string = remove_repeated_alpha_chars(string)
    if remove_excess_whitespace:
        string = remove_excess_whitespace_from_string(string)
    if lowercase_characters:
        string = string.lower()
    

    return string