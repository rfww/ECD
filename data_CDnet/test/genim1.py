import os

lines = open('image.txt').readlines()

lines = map(lambda s: s.rstrip(s.split('/')[-1]), lines)
open('image_new.txt', 'w').write(os.linesep.join(lines))
