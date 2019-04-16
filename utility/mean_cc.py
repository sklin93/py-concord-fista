import re

file_name = '../data-utility/cc.txt'
cc = []
pval = []
rms = []
number_regex = re.compile('-?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *-?\ *[0-9]+)?')

for line in open(file_name):
    if line[0:2] == 'cc':
        vals = [float(x) for x in re.findall(number_regex, line)]
        cc.append(vals[0])
        pval.append(vals[1])
    if line[0:3] == 'rms':
        vals = [float(x) for x in re.findall(number_regex, line)]
        rms.append(vals[0])

print(sum(cc)/float(len(cc)))
print(sum(pval)/float(len(pval)))
print(sum(rms)/float(len(rms)))

