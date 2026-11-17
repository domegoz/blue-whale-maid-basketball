"""Build the complete take from the reference art and recorded motion tracks."""
from pathlib import Path
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
ART = ROOT / '制作'
TAKE = ART / '篮球舞复刻'
RIG = ROOT / '骨骼动画'
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
sys.path.insert(0, str(ART))


def run(script, *args):
    print(f'运行 {script.name}', flush=True)
    subprocess.run([sys.executable, '-X', 'utf8', str(script), *args], cwd=ROOT, check=True)


def main():
    RIG.mkdir(exist_ok=True)
    run(ART / 'model.py')
    run(ART / 'render.py')
    run(ART / 'turntable.py')
    run(ART / 'viewer.py')
    run(ART / 'rig_animation.py')
    run(ART / 'preview_rig.py', '--check')
    import shoulder_dance
    base_dir = RIG / '篮球尺寸校验'
    base_dir.mkdir(exist_ok=True)
    base = base_dir / '大篮球_直径1.75倍.glb'
    shoulder_dance.write(base, radius=shoulder_dance.RAD * 1.75)
    from render_shoulder_dance import main as preview
    import render_shoulder_dance
    render_shoulder_dance.GLB = base
    render_shoulder_dance.MP4 = base_dir / '篮球1.75倍_4秒试渲染.mp4'
    preview()
    run(TAKE / 'retarget.py')
    run(TAKE / 'render_take.py')
    run(ROOT / '核对成片.py')


if __name__ == '__main__':
    main()
