
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.table import Table

from celluloid import Camera



def get_display_color(ndvi, lst):
    return "gray" if ndvi > 0.4 else "green"


def visited_states(item, visited):
    return (item[0], item[1]) in visited

def agent_current(item, current):
    return  (item[0], item[1]) == current

def render_current(map, current, WIDTH, HEIGHT, visited=[]):
    fig, ax = plt.subplots()
    ax.set_axis_off()
    tb = Table(ax, bbox=[0, 0, 1, 1])
    tb.set_fontsize(30)
    width, height = 1.0 / WIDTH, 1.0 / HEIGHT

    for item in map:
        #print("see {} - {} - {} - {}".format(item[0], item[1], item[2], item[3]))
        if visited_states(item, visited) :
            text = ""
            if agent_current(item, current) :
                text = "C"
            tb.add_cell(item[0], item[1], width, height, text=text
                        , loc="center", facecolor="red")
        else:
            tb.add_cell(item[0],item[1], width, height, text=""
                        , loc="center", facecolor= get_display_color(item[2], item[3]))

    ax.add_table(tb)

    plt.show()
    #plt.close()
    return fig

def save_animation(frames, filename):
    fig = plt.figure()
    camera = Camera(fig)
    for f in frames:
        plt.imshow(f)
        camera.snap()
    animation = camera.animate()
    animation.save(filename+'.gif', writer='PillowWriter', fps=2)