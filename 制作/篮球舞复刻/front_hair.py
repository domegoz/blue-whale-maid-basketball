"""Pinned world-space guides for the three fringe lobes, cheek curls and ahoge."""
import numpy as np
from scipy.interpolate import LinearNDInterpolator

from garment_dynamics import solve
from rig_animation import read_accessor, smooth


def simulate_front(doc,buf,head_pose,t,rig):
    origin,head=head_pose
    matrix=head.as_matrix()
    head_rest=rig.point('Head')
    dt=t[1]-t[0]
    face=doc['meshes'][0]['primitives'][0]['attributes']
    skin=read_accessor(doc,buf,face['POSITION'])
    uv=read_accessor(doc,buf,face['TEXCOORD_0'])[:,0]
    front=np.abs((uv-uv.min())/np.ptp(uv)-.5)<.232
    skin_depth=LinearNDInterpolator(skin[front,:2],skin[front,2])
    def chain(points,k,limit):
        n=len(points)
        goals=np.einsum('tij,nj->tni',matrix,points-head_rest)+origin[:,None]
        edges=[(i,i+step) for step in (1,2) for i in range(n-step)]
        strength=[.91]*(n-1)+[.29]*(n-2)
        along=np.linspace(0,1,n)
        stiffness=k*(1-.55*along)
        stiffness[:2]=0
        particles=solve(goals,np.asarray(edges),np.asarray(strength),stiffness,
                        limit*along**1.35,np.full(n,-5.),np.zeros((len(t),0,2,3)),np.zeros(0),dt)
        return np.einsum('tji,tnj->tni',matrix,particles-goals)
    def basis(delta,count):
        flat=delta.reshape(len(t),-1)
        _,_,modes=np.linalg.svd(flat,full_matrices=False)
        modes=modes[:count]
        coeff=flat@modes.T
        scale=np.maximum(1.e-7,np.max(np.abs(coeff),axis=0))
        return (scale[:,None]*modes).reshape(count,*delta.shape[1:]),coeff/scale
    def register(modes,along):
        idx=np.clip(along,0,1)*(modes.shape[1]-1)
        lo=idx.astype(int);hi=np.minimum(lo+1,modes.shape[1]-1)
        return modes[:,lo]*(1-(idx-lo))[None,:,None]+modes[:,hi]*(idx-lo)[None,:,None]
    result={}
    for mesh in doc['meshes']:
        name=mesh['name']
        if not name.startswith(('Cheek curl','Crown cowlick','Bangs ·')): continue
        attr=mesh['primitives'][0]['attributes']
        p=read_accessor(doc,buf,attr['POSITION']).copy()
        x,y,z=p.T
        if name.startswith('Bangs'):
            centers=[-.38,-.015,.36]
            ends=[1.025,1.227,1.017]
            weights=np.exp(-.5*((x[:,None]-centers)/[.129,.133,.125])**2)
            weights/=weights.sum(1,keepdims=True)
            guide=[];coordinates=[]
            for i,(cx,end) in enumerate(zip(centers,ends)):
                ysample=np.linspace(1.91,end,7)
                xsample=np.linspace(cx*.63,cx,7)
                points=[]
                for xx,yy in zip(xsample,ysample):
                    near=np.argsort(((x-xx)/.92)**2+(y-yy)**2)[:15]
                    points.append([xx,yy,np.percentile(z[near],77)])
                guide.append(chain(np.array(points),[246,300,220][i],[.098,.073,.097][i]))
                coordinates.append((1.91-y)/(1.91-end))
            modes,coeff=basis(np.stack(guide,axis=1),7)
            free=1-smooth(1.20,1.78,y)
            # Open the eye line and tip the centre tuft forward, with a fixed crown.
            ax=free*(.021*np.tanh(x/.19)+.008*weights[:,1])
            ay=free*(.021+.029*weights[:,1])
            az=free*(.014+.012*weights[:,1])
            p=p+np.column_stack((ax,ay,az))
            offsets=sum(register(modes[:,i],coordinates[i])*weights[None,:,i,None] for i in range(3))
            # Slide the fringe along the face. Limit the depth swing in the eye band.
            dx,dy=.005,.005
            sx=np.clip(np.nan_to_num((skin_depth(x+dx,y)-skin_depth(x-dx,y))/(2*dx)),-1.7,1.7)
            sy=np.clip(np.nan_to_num((skin_depth(x,y+dy)-skin_depth(x,y-dy))/(2*dy)),-1.7,1.7)
            band=(1-smooth(1.53,1.76,y))*smooth(.93,1.03,y)
            offsets[:,:,2]=offsets[:,:,2]*(1-.76*band)+band*(sx*offsets[:,:,0]+sy*offsets[:,:,1])
            # Retain a gap between the visible front surface and the skin even at maximal inward swing.
            face_z=skin_depth(x,y)
            over=np.isfinite(face_z)&(z>=np.nan_to_num(face_z,nan=9.)-.010)
            gap=np.minimum(.035,np.maximum(0,np.nan_to_num(face_z,nan=-9.)+.035-p[:,2]))
            p[:,2]+=gap*over*free
        else:
            uv=read_accessor(doc,buf,attr['TEXCOORD_0'])[:,1]
            along=(uv-uv.min())/np.ptp(uv)
            rings=np.unique(along)
            mid=np.array([p[along==v].mean(axis=0) for v in rings])
            cow=name=='Crown cowlick'
            sample=np.linspace(0,1,9 if cow else 7)
            points=np.stack([np.interp(sample,rings,axis) for axis in mid.T],axis=1)
            if cow:
                edit=np.column_stack((-.017*sample**2,.031*np.sin(np.pi*sample)**2,.008*sample))
            else:
                side=np.sign(x.mean())
                edit=np.column_stack((side*.025*sample**2,.009*sample**2,.041*sample**1.6))
            points+=edit
            p=p+np.stack([np.interp(along,sample,a) for a in edit.T],axis=1)
            movement=chain(points,118 if cow else 162,.158 if cow else .117)
            if not cow:
                movement[:,:,0]*=.68
                movement[:,:,2]*=.81
            modes,coeff=basis(movement,5)
            offsets=register(modes,along)
        assert np.all(np.isfinite(offsets)) and np.all(np.isfinite(p))
        result[name]=(p,offsets,coeff)
    return result
