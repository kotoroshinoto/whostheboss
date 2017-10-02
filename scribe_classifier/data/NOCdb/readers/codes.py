from csv import reader
import pickle


class CodeRecord:
    """This class represents a single entry in the NOC codefile"""
    def __init__(self, code: str, desc: str):
        """initialize instance using code and description"""
        self.code = code  # type: str
        self.desc = desc  # type: str

    def get_level(self):
        """get the level of this code"""
        return len(self.code)


class CodeSet:
    """this class represents a set of codes, typically reading all of them from the NOC codefile"""
    def __init__(self):
        """initializes with an empty dictionary and no emptyset label"""
        self.codes = dict()  # type: Dict[str, CodeRecord]
        self.emptyset_label = None

    def add_emptyset(self, emptyset_label: str ='NA', desc:str='Unknown'):
        """adds an emptyset label to the set, for convenience in getting code vectors"""
        if self.emptyset_label is not None:
            raise ValueError("Already have an emptyset: ", self.emptyset_label)
        self.emptyset_label = emptyset_label
        self.codes[self.emptyset_label] = CodeRecord(code=emptyset_label, desc=desc)

    def get_num_children(self, code: str):
        """calculate the number of child codes for given code"""
        if code == self.emptyset_label:
            return 0
        numchild = 0
        level = len(code)
        for code_key in self.codes:
            if code_key == self.emptyset_label:
                numchild += 1
                continue
            if len(code_key) != level + 1:
                continue
            short_code = code_key[0:level]
            if short_code == code:
                numchild += 1
        return numchild

    def get_children(self, code: str):
        """get the child codes of given code"""
        if code == self.emptyset_label:
            return []
        children = []
        level = len(code)
        for code_key in self.codes:
            if code_key == self.emptyset_label:
                children.append(self.codes[self.emptyset_label])
            if len(code_key) != level + 1:
                continue
            short_code = code_key[0:level]
            if short_code == code:
                children.append(code_key)
        return children

    def add_code(self, coderecord: 'CodeRecord'):
        """add a code record to the code set"""
        if coderecord.code not in self.codes:
            self.codes[coderecord.code] = coderecord
        else:
            raise ValueError("This code is already added")

    @staticmethod
    def parse_code_column(code_text) -> 'List[str]':
        """reads in codes from the CSV file"""
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
        """reads all codes from given file"""
        levels = reader(open(filename), dialect='excel-tab')
        if toss_first_line:
            first_line = next(levels)
        for entry in levels:
            codes = CodeSet.parse_code_column(entry[0])
            for code in codes:
                coderecord = CodeRecord(code, entry[1])
                self.add_code(coderecord)

    def get_codes_for_level(self, target_level=1):
        """get all codes for a given target level"""
        codes = list()
        for code_key in self.codes:  # type: CodeRecord
            code = self.codes[code_key]  # type: CodeRecord
            if code_key == self.emptyset_label:
                codes.append(code.code)
            elif code.get_level() == target_level:
                codes.append(code.code)
        return codes

    def get_codes_for_fitting_multi_level(self, target_level=1) -> 'Dict[int, List[str]]':
        """constructs a dictionary for use in multi-level model fitting"""
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

    def save_as_pickle(self, file, is_path=False):
        """save as pickle"""
        if is_path:
            handle = open(file, 'wb')
        else:
            handle = file
        pickle.dump(self, handle)
        handle.close()

    @staticmethod
    def load_from_pickle(file, is_path=False) -> 'CodeSet':
        """load from a saved pickle"""
        if is_path:
            handle = open(file, 'rb')
        else:
            handle = file
        ds = pickle.load(handle)
        handle.close()
        return ds
