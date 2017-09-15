#!/usr/bin/env python

from csv import reader

handle = open('UTF8/122372_Code.csv', 'r')
rdr = reader(handle)
first_line = next(rdr)
print("\t".join([first_line[0], first_line[1], first_line[3], first_line[5]]))
filestr = 'code_lvl_%d'
level_handles = []
for i in range(1,5):
    print("%d" % i)
    level_handles.append(open(filestr % i,'w'))
for entry in rdr:
    if entry[1].rstrip().lstrip() == "":
        continue
    kept_values=[entry[0], entry[1], entry[3], entry[5]]
    print("\t".join(kept_values), file=level_handles[int(entry[5]) - 1])
handle.close()
