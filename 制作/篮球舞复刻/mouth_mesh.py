"""Fit an expressive mouth patch to the painted face and bind it to the head."""
import io

import cv2
import numpy as np
from PIL import Image
from scipy.interpolate import LinearNDInterpolator

from rig_animation import read_accessor,smooth


def make_mouth(doc,buf,acc,nodes):
    prim=doc['meshes'][0]['primitives'][0]
    attr=prim['attributes']
    face=read_accessor(doc,buf,attr['POSITION'])
    uv=read_accessor(doc,buf,attr['TEXCOORD_0'])
    front=np.abs((uv[:,0]-uv[:,0].min())/np.ptp(uv[:,0])-.5)<.22
    xy=np.column_stack((403+300*face[:,0],695-300*face[:,1]))
    lookup=LinearNDInterpolator(xy[front],np.column_stack((face[front,2],uv[front])))
    image=doc['images'][0]
    view=doc['bufferViews'][image['bufferView']]
    start=view.get('byteOffset',0)
    atlas=np.array(Image.open(io.BytesIO(buf[start:start+view['byteLength']])).convert('RGB'))
    h,w=atlas.shape[:2]
    # The original neutral mouth is a few painted pixels. The patch below
    # supplies its outline in every pose including the resting pose.
    xx,yy=np.meshgrid(np.linspace(390,404,50),np.linspace(362,373,40))
    probe=lookup(np.column_stack((xx.ravel(),yy.ravel())))[:,1:]*[w,h]-.5
    hull=cv2.convexHull(np.rint(probe).astype('i4'))
    x0,y0,ww,hh=cv2.boundingRect(hull)
    x0-=6;y0-=6;ww+=12;hh+=12
    patch=atlas[y0:y0+hh,x0:x0+ww].copy()
    mask=np.zeros((hh,ww),'u1')
    cv2.fillConvexPoly(mask,hull.reshape(-1,2)-[x0,y0],255)
    atlas[y0:y0+hh,x0:x0+ww]=cv2.inpaint(patch,mask,3,cv2.INPAINT_TELEA)
    # The basketball atlas strip has unused space to the right of the ball.
    tx,ty=640,h-170
    tw,th=112,90
    for mesh in doc['meshes']:
        used=read_accessor(doc,buf,mesh['primitives'][0]['attributes']['TEXCOORD_0'])*[w,h]
        assert used[:,0].max()<tx-4 or used[:,0].min()>tx+tw+4 or used[:,1].max()<ty-4 or used[:,1].min()>ty+th+4
    ax,ay=np.meshgrid(np.linspace(-1,1,tw),np.linspace(0,1,th))
    upper=np.array([86,47,66]);lower=np.array([161,75,92])
    colour=upper+(lower-upper)*ay[:,:,None]
    tongue=np.exp(-((ax-.02)/.77)**4-((ay-.88)/.29)**4)
    colour+=tongue[:,:,None]*[50,38,38]
    edge=np.maximum(1-smooth(.0,.14,ay),np.maximum(smooth(.89,1.,ay),smooth(.84,1.,abs(ax))))
    colour=colour*(1-edge[:,:,None]) + [109,58,74]*edge[:,:,None]
    tile=np.uint8(np.clip(colour,0,255))
    atlas[ty-4:ty+th+4,tx-4:tx+tw+4]=np.pad(tile,((4,4),(4,4),(0,0)),mode='edge')
    packed=io.BytesIO();Image.fromarray(atlas).save(packed,format='PNG',compress_level=6)
    png=packed.getvalue()
    old_size=(view['byteLength']+3)//4*4
    block=png+bytes(-len(png)%4)
    buf[start:start+old_size]=block
    for other in doc['bufferViews']:
        if other.get('byteOffset',0)>=start+old_size:
            other['byteOffset']+=len(block)-old_size
    view['byteLength']=len(png)
    nx,ny=47,11
    u,v=np.meshgrid(np.linspace(-1,1,nx),np.linspace(0,1,ny))
    u,v=u.ravel(),v.ravel()
    cap=np.maximum(.018,1-u*u)**.65
    def profile(width,top,bottom,corner,asym=0):
        px=396.5+width*u
        py=367.8+corner*u*u+asym*u+((v-1)*top+v*bottom)*cap
        z=lookup(np.column_stack((px,py)))[:,0]
        return np.asarray(np.column_stack(((px-403)/300,(695-py)/300,z+.0025)),dtype='f4')
    neutral=profile(5.1,1.0,.85,.40)
    targets=[profile(8.0,1.05,1.22,.58,-1.0)-neutral,
             profile(15.8,1.55,6.5,-4.65,-.38)-neutral,
             profile(7.7,5.7,6.5,-.6)-neutral,
             profile(15.1,4.9,10.9,-.35)-neutral,profile(18.1,2.4,3.8,-1.9)-neutral,
             profile(6.4,4.1,8.6,-.18)-neutral,profile(10.2,.52,.44,-.65)-neutral,
             profile(8.9,1.,.85,-4.2,-.85)-neutral]
    idx=np.arange(nx*ny).reshape(ny,nx)[:-1,:-1].ravel()
    triangles=np.vstack((np.column_stack((idx,idx+nx,idx+1)),
                         np.column_stack((idx+1,idx+nx,idx+nx+1))))
    tangent=(lookup(np.column_stack((396.5+5.1*u+.12,367.8+v)))[:,0]-
             lookup(np.column_stack((396.5+5.1*u-.12,367.8+v)))[:,0])*300/.24
    normal=np.column_stack((-tangent,np.zeros_like(u),np.ones_like(u)))
    normal/=np.linalg.norm(normal,axis=1,keepdims=True)
    tu=(tx+(.5*u+.5)*(tw-1)+.5)/w
    tv=(ty+v*(th-1)+.5)/h
    joints=np.zeros((len(u),4),'u1');joints[:,0]=doc['skins'][0]['joints'].index(nodes['Head'])
    weights=np.zeros((len(u),4),'f4');weights[:,0]=1
    attributes=dict(POSITION=acc(neutral,'VEC3',True),NORMAL=acc(normal,'VEC3'),
                    TEXCOORD_0=acc(np.column_stack((tu,tv)),'VEC2'),
                    JOINTS_0=acc(joints,'VEC4',component=5121),WEIGHTS_0=acc(weights,'VEC4'))
    pr=dict(attributes=attributes,indices=acc(triangles.ravel(),'SCALAR',component=5123),material=prim['material'])
    name='Mouth expression'
    mesh=len(doc['meshes']);node=len(doc['nodes'])
    doc['meshes'].append(dict(name=name,primitives=[pr]))
    doc['nodes'].append(dict(name=name,mesh=mesh,skin=0))
    doc['scenes'][doc['scene']]['nodes'].append(node)
    nodes[name]=node
    return targets
