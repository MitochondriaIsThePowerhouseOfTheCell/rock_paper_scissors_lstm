import codecs
import unicodedata

f = codecs.open("02 - The Two Towers.txt", "r", encoding='ISO-8859-1')
dat = f.read()
f.close()

sequence = []
for item in dat:
    num = ord(item) % 3
    if num == 0:
        sequence.append('r')
    elif num == 1:
        sequence.append('p')
    else:
        sequence.append('s')

f = open("converted_TTT.txt", "w")
for item in sequence:
    f.write(item)
f.close()
