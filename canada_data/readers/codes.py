from _csv import reader


class CodeRecord:
    def  __init__(self, code: str, desc: str):
        self.code = code  # type: str
        self.desc = desc  # type: str

    def get_level(self):
        return len(self.code)

    # def is_compatible(self, other_code: str, target_level: int):
    #     if self.get_level() < target_level:
    #         return False
    #     return self.code[0:target_level] == other_code[0:target_level]


class AllCodes:
    def __init__(self):
        self.codes = dict()  # type: Dict[str, CodeRecord]

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

    def add_codes_from_file(self, filename):
        levels = reader(open(filename), dialect='excel-tab')
        first_line = next(levels)
        for entry in levels:
            codes = AllCodes._parse_code_column(entry[0])
            for code in codes:
                coderecord = CodeRecord(code, entry[1])
                self.add_code(coderecord)
