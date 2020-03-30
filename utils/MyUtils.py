import matplotlib.pyplot as plt


def show_plant(imgs, each_row=2):
    row = ((len(imgs) + each_row - 1)) // each_row
    fig = plt.figure(figsize=(8, 6), dpi=100)
    axes = fig.subplots(row, each_row)
    for i, ax in enumerate(axes.flatten()):
        ax.axis('off')
        ax.imshow(imgs[i])
    plt.show()
