import os

lines = open('image2.txt').readlines()

lines = map(lambda s: s.rstrip(s.split('/')[-1]), lines)
open('image.txt', 'w').write(os.linesep.join(lines))
