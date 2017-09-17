from _csv import reader


class CodeRecord:
    def  __init__(self, code: str, desc: str):
        self.code = code.lower()
        self.desc = desc

    def get_level(self):
        return len(self.code)


class AllCodes:
    def __init__(self):
        self.codes = dict

    def add_code(self, coderecord: 'CodeRecord'):
        if coderecord.code not in self.codes:
            self.codes[coderecord.code] = coderecord
        else:
            raise ValueError("This code is already added")

    @staticmethod
    def _parse_code_column(code_text) -> 'List[str]':
        codes = code_text.split('-')  # type: List[str]
        if len(codes) == 1:
            return codes
        elif len(codes) == 2:
            ret_codes = []
            first = codes[0]
            last = codes[1]
            beginning = first[:-1]
            last_digit_first = int(first[-1:])
            last_digit_last = int(last[-1:])
            for j in range(last_digit_first, last_digit_last + 1):
                ret_codes.append("%s%d" % (beginning, j))
            return ret_codes
        else:
            raise ("unparseable label detected: %s" % code_text)

    @classmethod
    def from_file(cls, filename) -> 'AllCodes':
        allcodes = cls()
        levels = reader(open(filename), dialect='excel-tab')
        first_line = next(levels)
        for entry in levels:
            codes = AllCodes._parse_code_column(entry[0])
            for code in codes:
                coderecord = CodeRecord(code, entry[1])
                allcodes.add_code(coderecord)
        return allcodes


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
            raise ("unparseable label detected: %s" % entry[0])
    return level_codes