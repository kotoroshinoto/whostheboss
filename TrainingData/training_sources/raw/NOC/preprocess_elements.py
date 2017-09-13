#!/usr/bin/env python

from csv import reader

handle = open('UTF8/122372_Element.csv', 'r')
rdr = reader(handle)
first_line = next(rdr)
print("\t".join([first_line[0],first_line[1],first_line[3],first_line[4]]))
for entry in rdr:
    if int(entry[3]) != 19:
        continue
    if entry[1].rstrip().lstrip() == "":
        continue
    kept_values=[entry[0],entry[1],entry[3],entry[4]]
    print("\t".join(kept_values))
handle.close()