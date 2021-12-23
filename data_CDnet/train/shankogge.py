import os

lines = open('label_train.txt').readlines()
lines = map(lambda s: s.rstrip(), lines)
open('label.txt', 'w').write(os.linesep.join(lines))
