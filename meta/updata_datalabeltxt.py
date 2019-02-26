import os
import glob

files = glob.glob('area?_data_label.txt')
print(files)
for f in files:
    fopen = open(f, 'r')
    
    new_lines = [] 
    for line in fopen.readlines():
        new_line = line.replace('stanford_indoor3d', 'stanford_indoor3d_ins.sem')
        print(new_line)
        new_lines.append(new_line)

    fwrite = open(f, 'w')
    
    fwrite.writelines(new_lines)


    
