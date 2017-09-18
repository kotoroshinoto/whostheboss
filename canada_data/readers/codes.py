from _csv import reader


class CodeRecord:
    def  __init__(self, code: str, desc: str):
        self.code = code  # type: str
        self.desc = desc  # type: str

    def get_level(self):
        return len(self.code)


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

    def add_codes_from_file(self, filename, toss_first_line=False):
        levels = reader(open(filename), dialect='excel-tab')
        if toss_first_line:
            first_line = next(levels)
        for entry in levels:
            codes = AllCodes._parse_code_column(entry[0])
            for code in codes:
                coderecord = CodeRecord(code, entry[1])
                self.add_code(coderecord)

    def get_codes_for_level(self, target_level=1):
        codes = list()
        for code_key in self.codes:  # type: CodeRecord
            code = self.codes[code_key]
            if code.get_level() == target_level:
                codes.append(code.code)
        return codes

    def get_codes_for_fitting_multi_level(self, target_level=1) -> 'Dict[int, List[str]]':
        codes_by_level = dict()  # type: Dict[int, List[str]]
        codes_by_level[0] = [""]
        for code_key in self.codes:  # type: CodeRecord
            code = self.codes[code_key]
            code_level = code.get_level()
            if code_level < target_level:
                # print("code_key: %s" % code_key)
                if code_level not in codes_by_level:
                    codes_by_level[code_level] = list()
                codes_by_level[code_level].append(code.code)
        return codes_by_level