
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.table import Table

from cv2 import VideoWriter, VideoWriter_fourcc
import matplotlib.animation as animation



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
    plt.close()


'''

TODO: trying to make the rendered images into a video

from dataset_generator import data_set, WIDTH, HEIGHT
import random
fig = render_current(data_set, (random.randint(0,9), random.randint(0,9)),WIDTH, HEIGHT, visited=[(0,0), (0,1), (0,3)])

img = np.array(fig.canvas.renderer._renderer)

w,h = fig.canvas.get_width_height()
buf = np.fromstring(fig.canvas.to_string_argb(), dtype=np.uint8)
buf.shape = (w,h,4)

#convert back to rgb mode
buf = np.roll(buf, 3, axis=2)
print(buf)

print(img)

width = 1280
height = 720
FPS = 1
seconds = 10



fourcc = VideoWriter_fourcc(*"MP42")
video = VideoWriter("./plot_animation.avi", fourcc, float(FPS), (width, height))

for i in range(FPS*seconds):
    video.write(img)
    print("iteration: {} ".format(i))
video.release()


imgs = []
imgs.append([plt.imshow(img)])
fig = plt.figure()
Writer = animation.writers["ffmpeg"]
writer = Writer(fps=FPS, metadata=dict(artist="Me"), bitrate=18000)
ani = animation.ArtistAnimation(fig, imgs, interval=FPS, blit=True, repeat_delay=0)
ani.save("plot_animation.mp4", writer=writer)

print(img.shape)

plt.imshow(img)
#plt.show()

'''