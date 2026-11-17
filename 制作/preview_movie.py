"""Encode RGB preview frames while preserving Unicode paths on Windows."""
from fractions import Fraction
from pathlib import Path
import av
import numpy as np

def write_movie(destination, images, size, fps):
    path=Path(destination)
    temp=path.with_suffix('.building.mp4')
    with av.open(str(temp),'w',options={'movflags':'+faststart'}) as out:
        track=out.add_stream('libx264',rate=fps)
        track.width,track.height=size
        track.pix_fmt='yuv420p'
        track.options={'preset':'veryfast','crf':'17'}
        for i,img in enumerate(images):
            frame=av.VideoFrame.from_ndarray(np.asarray(img),format='rgb24')
            frame.pts=i
            frame.time_base=Fraction(1,fps)
            out.mux(track.encode(frame))
        out.mux(track.encode(None))
    temp.replace(path)
