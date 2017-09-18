import re
import sys
import os

list_extractor = re.compile("^unique (.+):\[(.+)\]$")
split_point = re.compile(", u['\"]")


def process_file(filename):
    retval = []
    cityfile = open("exported/%s" % filename)
    line = cityfile.readline()
    listmatch = list_extractor.match(line)
    if not listmatch:
        return []
    listname = listmatch.group(1)
    print("extracting %s data" % listname)
    content = listmatch.group(2)
    split_obj = split_point.split(content)
    for item in split_obj:
        if item == 'None' or (item.startswith('u') and (item[1] == '"' or item[1] == "'")):
            retval.append(item)
        else:
            retval.append("u%s%s" % (item[-1], item))
    return retval


def clean_file(filename):
    filecontent = process_file(filename)
    outfile = open('clean/%s' % filename, 'w')
    # outfile = sys.stdout
    print("\n".join(filecontent), file=outfile)
    outfile.close()


os.chdir("data/unique_labels")
clean_file("cities.txt")
clean_file("countries.txt")
clean_file("industries.txt")
clean_file("titles.txt")
