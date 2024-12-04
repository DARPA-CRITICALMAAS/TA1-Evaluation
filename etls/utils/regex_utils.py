"""
Author: Anastassios Dardas, PhD - Lead Geospatial Computing Engineer

Date Created: March 2024

Date Modified: March 2024

About: Generate Regex output from a string.
"""
import re
from difflib import SequenceMatcher
from typing import Union
from numpy import argmax


class GenRegex:

    def __init__(self, string):
        self.string    = string
        self.regex_str = ""
        self.gen_regex()

    def gen_regex(self):
        last_idx_type = -1
        current_type  = self.identify_char_type(self.string[0])
        current_type_set = True

        for idx, char in enumerate(self.string):
            last_idx_type += 1
            if ((self.identify_char_type(char) != current_type) or self.identify_char_type(char) == "SPECIAL"):
                self.set_regex_str(current_type, last_idx_type, idx)
                current_type = self.identify_char_type(char)
                last_idx_type = 0

                if (idx == len(self.string) - 1):
                    last_idx_type += 1
                    self.set_regex_str(current_type, last_idx_type, idx)

            if (idx == len(self.string) - 1 and current_type == self.identify_char_type(self.string[idx-1])):
                last_idx_type += 1
                self.set_regex_str(current_type, last_idx_type, idx)

    def acq_regex(self):
        return self.regex_str

    def set_regex_str(self, current_type, last_idx_type, idx):

        if current_type == "DIGIT":
            self.regex_str += ("\\d{"+str(last_idx_type)+"}")

        elif current_type == "CHARACTER":
            self.regex_str += ("\\w{"+str(last_idx_type)+"}")

        elif current_type == "WHITESPACE":
            self.regex_str += ("\\s{"+str(last_idx_type)+"}")

        elif current_type == "SPECIAL":
            self.regex_str += "[" + self.string[idx-1] + "]"

    def identify_char_type(self, char):

        if char.isnumeric():
            return "DIGIT"

        elif char.isalpha():
            return "CHARACTER"

        elif char.isspace():
            return "WHITESPACE"

        else:
            return "SPECIAL"


class ReMethods:
    def __init__(self):
        pass

    def re_replace(self, regex, replace_by, string):
        return re.sub(regex, replace_by, string)

    def re_replace_and_split(self, regex, split_by, string):
        return self.re_replace(regex=regex, replace_by=split_by, string=string).split(split_by)

    def sequence_match(self, str1, str2):
        return SequenceMatcher(None, str1, str2).ratio()

    def max_sequence_matcher(self, str1, str2_list: Union[str, list]):
        if isinstance(str2_list, list):
            ratio_list = [self.sequence_match(str1=str1, str2=str2) for str2 in str2_list]
            max_idx    = argmax(ratio_list)

            return [str2_list[max_idx], ratio_list[max_idx]]

        elif isinstance(str2_list, str):
            get_max = self.sequence_match(str1=str1, str2=str2_list)
            return [str2_list, get_max]


#GenRegex(string="asdfs902342*afs").regex_str