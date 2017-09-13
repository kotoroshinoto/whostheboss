#!/usr/bin/env python
from csv import reader
from typing import List,Dict
import sys

class TitleRecord:
    def __init__(self, code: str, title: str):
        self.code = code  # type: str
        self.title = title  # type: str

    def get_code(self, level: int):
        return self.code[0:level]

    def combine_text(self, code_dict:'Dict[str, str]', target_level:int) -> 'str':
        combined = dict()
        # target_code = self.get_code(target_level)
        # print(target_code)
        text = []
        for i in range(target_level, 4):
            text.append(code_dict[self.get_code(i)])
        text.append(self.title)
        return " ".join(text)


def read_levels(filename) -> 'Dict[str, str]':
    levels = reader(open(filename), dialect='excel-tab')
    first_line = next(levels)
    level_codes = dict()
    for entry in levels:
        codes = entry[0].split('-')  # type: List[str]

        if len(codes) == 1:
            level_codes[entry[0]] = entry[1]
        elif len(codes) == 2:
            first = codes[0]
            last = codes[1]
            beginning = first[:-1]
            last_digit_first = int(first[-1:])
            last_digit_last = int(last[-1:])
            for j in range(last_digit_first, last_digit_last + 1):
                jth_code = "%s%d" % (beginning, j)
                level_codes[jth_code] = entry[1]
        else:
            raise ("unparseable label detected: %s" % label)
    return level_codes


def read_titles(filename, target_level) -> 'List[TitleRecord]':
    tlvl = int(target_level)
    titles = reader(open(filename), dialect='excel-tab')
    records = []
    for title_record in titles:
        records.append(TitleRecord(title_record[0],title_record[1]))
    return records


def print_combined(level_codes: 'Dict[str, str]', title_records: 'List[TitleRecord]', target_level, target_file= sys.stdout):
    for trecord in title_records:
        print("%s\t%s" % (trecord.get_code(target_level), trecord.combine_text(level_codes, target_level)), file=target_file)


def print_uncombined_text_for_target_level(title_records: 'List[TitleRecord]', target_level, target_file = sys.stdout):
    for trecord in title_records:
        print("%s\t%s" % (trecord.get_code(target_level), trecord.title), file=target_file)


def main():
    target_level = 2
    level_codes = read_levels('all_codes')
    title_records = read_titles('all_examples', target_level)
    print_combined(level_codes, title_records, 2, open('combined_text_lvl_%d' % target_level, 'w'))
    print_uncombined_text_for_target_level(title_records, 2, open('uncombined_text_lvl_%d' % target_level, 'w'))


if __name__ == "__main__":
    main()