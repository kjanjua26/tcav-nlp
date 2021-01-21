"""
File to parse: 
Parse professions with [MASK] in a single txt file.
"""

from tqdm import tqdm
import re

output = "/Users/Janjua/Desktop/QCRI/Work/aux_classifier/aux_classifier/mask.txt"

def reformat():
    """Read the file and convert [his, her] to MASK."""
    
    to_mask = ['his', 'her']
    outlines = []
    fp = "/Users/Janjua/Desktop/QCRI/Work/aux_classifier/aux_classifier/handcrafted.ende.txt"

    with open(fp, 'r') as data:
        lines = data.readlines()
        for line in tqdm(lines):
            line = line.strip()
            line = line.split(' .|')[0]
            line = [re.sub(r"\b{}\b".format(word), "[MASK]", line) \
                for word in line.split(' ') if word in to_mask][0]
            outlines.append(line)
    data.close()

    unique = set(outlines)
    with open(output, 'w') as out:
        for line in tqdm(unique):
            out.write(line + "\n")
    out.close()

reformat()