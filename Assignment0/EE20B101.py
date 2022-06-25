"""
   Frequency counter of each word in the file `text.txt`
"""

import re  # regex/regual expressions

dict = {}
with open("text.txt", encoding="utf8") as file:
    for line in file:
        words = re.split("[^a-z]+", line.lower())
        for word in words:
            if word in dict:
                dict[word] += 1
            elif word != "":
                dict[word] = 1

print(sorted(dict.items(), key=lambda x: x[1], reverse=True))
