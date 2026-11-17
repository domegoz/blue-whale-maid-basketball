from shoulder_dance import GLB,MP4,ROOT,FPS,RAD
from preview_rig import Player
from render import render
from PIL import Image
import numpy as np,sys,subprocess,cv2

CLIP='Basketball Shoulder Dance'
CAM=3
TILT=3

def screenshot(model,t,size):
    data=model.frame(CLIP,t);img=np.array(render(data,CAM,TILT,size));p=data[0]
    w,h=size;scale=h/2.56;cx=np.array([0,1.14,-.005])
    a,e=np.deg2rad([CAM,TILT]);right=np.array([np.cos(a),0,-np.sin(a)])
    up=np.array([-np.sin(a)*np.sin(e),np.cos(e),-np.cos(a)*np.sin(e)])
    yy,xx=np.ogrid[:h,:w];shadow=np.zeros((h,w),dtype='f')
    for name in ('Mary Jane -1','Mary Jane 1','Basketball mesh'):
        g=p[model.groups[name]];x,y,z=(g.min(0)+g.max(0))*.5
        altitude=max(0,g[:,1].min());ground=np.array([x,0,z])-cx
        sx=w/2+ground@right*scale;sy=h/2-ground@up*scale
        k=(.13 if name.startswith('Mary') else .10)+altitude*.12
        opacity=(.12 if name.startswith('Mary') else .13)*np.exp(-4.2*altitude)
        shadow+=opacity*np.exp(-((xx-sx)/(scale*k))**2-((yy-sy)/(scale*(.012+altitude*.045)))**2)
    background=np.all(img==[250,249,247],axis=2)
    img[background]=(img[background]*(1-shadow[background,None])+np.array([105,97,118])*shadow[background,None]).astype('u1')
    return Image.fromarray(img)
def main():
    model=Player(GLB)
    if '--poses' in sys.argv:
        times=[.0,.35,.74,1.4,1.57,1.96,2.19,2.57,2.75,3.22]
        page=Image.new('RGB',(5*252,2*290))
        for i,t in enumerate(times):page.paste(screenshot(model,t,(416,478)).resize((252,290),Image.Resampling.LANCZOS),(i%5*252,i//5*290))
        page.save(ROOT/'制作'/'篮球舞动作核对.jpg');return
    w,h=720,800
    from preview_movie import write_movie
    import av
    imgs=(screenshot(model,t/FPS,(w,h)) for t in range(4*FPS))
    write_movie(MP4,imgs,(w,h),FPS)
    with av.open(str(MP4)) as check:
        assert sum(1 for frame in check.decode(video=0))==4*FPS

if __name__=='__main__':main()
