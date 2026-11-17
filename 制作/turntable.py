from render import ROOT,read,render
from preview_movie import write_movie

data=read()
size=(652,736)
frames=(render(data,step*360/144,2.5,size) for step in range(144))
write_movie(ROOT/'模型旋转展示.mp4',frames,size,24)
