
def read_image_class(path):
    print(path)
    image = []
    classi = []
    with open(path, 'r') as f:
        for line in f:
            
            image.append(line.strip().split(" ", 1)[0])
            classi.append(int(line.strip().split(" ", 1)[1]))
    f.close()
    return image, classi


def read_label(path):
    label = []

    with open(path, 'r') as f:
        for line in f:
            label.append(line.strip())
    f.close()
    return label
