from pathlib import Path
import numpy as np
import json,struct,io,sys, math
from PIL import Image,ImageDraw,ImageFont
import cv2
from numba import njit
ROOT=Path(__file__).resolve().parents[1]
def read():
    b=(ROOT/'蓝鲸女仆.glb').read_bytes();magic,version,total=struct.unpack_from('<4sII',b);assert magic==b'glTF' and total==len(b) and version==2
    l=struct.unpack_from('<I',b,12)[0];j=json.loads(b[20:20+l]);binp=memoryview(b)[28+l:]
    def view(i):
        v=j['bufferViews'][i];return binp[v['byteOffset']:v['byteOffset']+v['byteLength']]
    def acc(i):
        t=j['accessors'][i];k={'SCALAR':1,'VEC2':2,'VEC3':3}[t['type']];a=np.frombuffer(view(t['bufferView']),{5126:'<f4',5123:'<u2'}[t['componentType']]).reshape(-1,k);assert np.isfinite(a).all();return a
    tex=np.array(Image.open(io.BytesIO(view(j['images'][0]['bufferView']))).convert('RGB'))
    P=[];N=[];U=[];F=[];ids=[];cnt=0
    for i,m in enumerate(j['meshes']):
        g=m['primitives'][0];a=g['attributes'];p=acc(a['POSITION']);n=acc(a['NORMAL']);u=acc(a['TEXCOORD_0']);f=acc(g['indices']).reshape(-1,3).astype('i')
        assert f.max()<len(p) and np.all((u>=0)&(u<=1)) and np.allclose(np.linalg.norm(n,axis=1),1,atol=.001)
        P.append(p);N.append(n);U.append(u);F.append(f+cnt);ids.extend([i]*len(f));cnt+=len(p)
    return np.vstack(P),np.vstack(N),np.vstack(U),np.vstack(F),np.array(ids),tex
@njit(cache=True)
def paint(buf,zb,idb,pts,uv,col,F,ids,tex):
    H,W=zb.shape;th,tw,_=tex.shape
    for fi in range(len(F)):
        i,j,k=F[fi];x0,y0,z0=pts[i];x1,y1,z1=pts[j];x2,y2,z2=pts[k]
        d=(y1-y2)*(x0-x2)+(x2-x1)*(y0-y2)
        if abs(d)<1e-6:continue
        for yy in range(max(0,int(np.floor(min(y0,y1,y2)))),min(H,int(np.ceil(max(y0,y1,y2)))+1)):
            y=yy+.5
            for xx in range(max(0,int(np.floor(min(x0,x1,x2)))),min(W,int(np.ceil(max(x0,x1,x2)))+1)):
                x=xx+.5
                a=((y1-y2)*(x-x2)+(x2-x1)*(y-y2))/d;b=((y2-y0)*(x-x2)+(x0-x2)*(y-y2))/d;c=1-a-b
                if min(a,b,c)<-1e-6:continue
                z=z0*a+z1*b+z2*c
                if z<zb[yy,xx]:continue
                zb[yy,xx]=z;idb[yy,xx]=ids[fi]
                tx=(a*uv[i,0]+b*uv[j,0]+c*uv[k,0])*tw-.5;ty=(a*uv[i,1]+b*uv[j,1]+c*uv[k,1])*th-.5
                ix=int(tx);iy=int(ty);fx=tx-ix;fy=ty-iy
                shade=a*col[i]+b*col[j]+c*col[k]
                for channel in range(3):
                    cr=(tex[iy,ix,channel]*(1-fx)+tex[iy,ix+1,channel]*fx)*(1-fy)+(tex[iy+1,ix,channel]*(1-fx)+tex[iy+1,ix+1,channel]*fx)*fy
                    buf[yy,xx,channel]=min(255,max(0,cr*shade))
def render(data,angle,tilt,size=(760,860)):
    P,N,U,F,ids,T=data; w,h=size
    a,e=np.deg2rad([angle,tilt]);view=np.array([np.sin(a)*np.cos(e),np.sin(e),np.cos(a)*np.cos(e)])
    right=np.array([np.cos(a),0,-np.sin(a)]);up=np.cross(view,right)
    ctr=np.array([0,1.14,-.005])
    p=(P-ctr)@np.vstack([right,-up,view]).T;px=h/2.56;p[:,:2]*=px;p[:,:2]+=[w*.5,h*.5]
    lights=right*(-.40)+up*.58+view*.70
    col=.91+.11*np.maximum(0,N@lights)+.025*np.maximum(0,N@up)
    buf=np.full((h,w,3),[250,249,247],np.uint8);zb=np.full((h,w),-30.,np.float32);ob=np.full((h,w),-1,'i')
    paint(buf,zb,ob,p.astype('f'),U,col.astype('f'),F,ids,T)
    # Subtle ink only at true depth discontinuities, with gentle contact shading.
    near=cv2.dilate(zb,np.ones((3,3),np.uint8));far=cv2.erode(zb,np.ones((3,3),np.uint8));mask=ob>=0
    edges=np.clip((zb-far-.037)/.12,0,1)
    ink=edges*.34*mask
    ao=np.zeros_like(zb)
    for dx,dy in [(5,0),(-5,0),(0,5),(0,-5),(-4,-4),(4,4),(4,-4),(-4,4)]:
        diff=np.roll(zb,(dy,dx),axis=(0,1))-zb;ao+=np.clip((diff-.02)*4,0,1)*np.clip(1-diff/1.8,0,1)
    img=buf.astype('f')*(1-.012*ao[...,None]*mask[...,None])
    img=img*(1-ink[...,None])+np.array([44,40,66])*ink[...,None]
    return Image.fromarray(np.uint8(np.clip(img,0,255)))
def sheet(data):
    panels=[render(data,a,0,(620,800)) for a in (0,90,180)]
    sheet=Image.new('RGB',(1650,740),'#faf9f7')
    for i,im in enumerate(panels):
        im=im.resize((550,710),Image.Resampling.LANCZOS);sheet.paste(im,(i*550,5))
    d=ImageDraw.Draw(sheet)
    try:
        font=ImageFont.truetype('msyh.ttc',17)
        labels=['正面','侧面','背面']
    except OSError:
        font=ImageFont.load_default(size=17)
        labels=['Front','Side','Back']
    for i,t in enumerate(labels):d.text((275+i*550,712),t,font=font,fill=(91,88,105),anchor='mm')
    sheet.save(ROOT/'模型三视图.png')
if __name__=='__main__':
    data=read();sheet(data)
    render(data,25,3,(800,930)).save(ROOT/'制作'/'视角检查.png')
