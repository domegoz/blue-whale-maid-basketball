"""Read the emitted video and GLB without regenerating their contents."""
from fractions import Fraction
from pathlib import Path
import sys

import av
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / '制作' / '篮球舞复刻'))
from retarget import CLIP, GLB, IMPACTS, RADIUS
from preview_rig import Player


def main():
    mp4 = GLB.with_name('蓝鲸女仆_自我介绍与篮球舞_1080p60.mp4')
    with av.open(str(mp4)) as container:
        video = container.streams.video[0]
        audio = container.streams.audio[0]
        assert (video.width, video.height, video.average_rate) == (1920, 1080, 60)
        duration = Fraction(493, 10)
        assert video.duration * video.time_base == duration
        assert len(audio.codec_context.layout.channels) == 2
        assert abs(audio.duration * audio.time_base - duration) < .023
        count = 0
        for frame in container.decode(video=0):
            assert abs(frame.time - count/60) < .0001
            count += 1
        assert count == 2958
    player = Player(GLB)
    for name, (ts, values) in player.morph_anim[CLIP].items():
        assert len(ts) == 2959 and abs(ts[-1] - 49.3) < .0001
        assert np.isfinite(values).all()
    weights = player.morph_anim[CLIP]['Mouth expression'][1]
    assert np.ptp(weights[:510, 3:7], axis=0).min() > .08
    for name in ('Bangs · sculpted parted fringe', 'Cheek curl 1', 'Cheek curl -1', 'Crown cowlick'):
        tracks = player.morph_anim[CLIP][name][1]
        assert np.abs(tracks).max() > .2
    for hit in IMPACTS:
        w, _ = player.matrices(CLIP, round(hit*30)/30)
        assert abs(w[player.names.index('Basketball'),1,3] - RADIUS) < .0001
    assert (ROOT/'旋转查看模型.html').is_file()
    print('通过核验：2958 帧、双声道音画时长、口型、四组发片及 11 次篮球触地。')
    print(mp4)
    print(GLB)


if __name__ == '__main__':
    main()
