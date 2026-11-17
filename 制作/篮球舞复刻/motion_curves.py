"""Smooth the dance in joint space and solve contacts at the final sampling rate."""
import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.ndimage import binary_closing, gaussian_filter1d
from scipy.spatial.transform import Rotation

from rig_animation import smooth


def unit(v):
    return v / np.maximum(np.linalg.norm(v,axis=-1,keepdims=True),1e-10)


def curve(v,t,dense,sigma):
    v=gaussian_filter1d(v,sigma,axis=0,mode='nearest')
    return CubicSpline(t,v,axis=0)(dense)


def orientation(rotation,t,dense,sigma):
    q=rotation.as_quat()
    flips=np.sum(q[1:]*q[:-1],axis=1)<0
    signs=np.cumprod(np.r_[1,np.where(flips,-1,1)])
    return Rotation.from_quat(unit(curve(q*signs[:,None],t,dense,sigma)))


def shortest(rest,direction):
    a=unit(np.asarray(rest))
    b=unit(direction)
    return Rotation.from_quat(unit(np.column_stack((np.cross(a,b),1+b@a))))


def intervals(mask):
    transitions=np.diff(np.r_[False,mask,False].astype(int))
    return zip(np.flatnonzero(transitions==1),np.flatnonzero(transitions==-1))


def plant(ankle,yaw,t):
    floor=.092
    target=ankle.copy()
    heading=yaw.copy()
    # Blends start before contact and release into the following step.
    def hold(start,end,fade,limit):
        region=(t>=start)&(t<=end)
        if not np.any(region):
            return
        weight=smooth(start-fade,start,t)*(1-smooth(end,end+fade,t))
        center=np.median(ankle[region][:,[0,2]],axis=0)
        shift=center-target[:,[0,2]]
        length=np.linalg.norm(shift,axis=1)
        shift*=np.minimum(1,limit/np.maximum(length,1e-7))[:,None]
        target[:,[0,2]]+=weight[:,None]*shift
        target[:,1]+=weight*(floor-target[:,1])
        angle=np.median(yaw[region])
        heading[:]+=weight*np.clip(angle-heading,-.65,.65)
    grounded=binary_closing(ankle[:,1]<floor+.030,structure=np.ones(5,bool))
    for lo,hi in intervals(grounded):
        if t[hi-1]-t[lo]>=.14:
            hold(t[lo],t[hi-1],.085,.12)
    for start,end in [(0,7.65),(8.80,11.40),(24.96,26.54),(47.38,49.30)]:
        hold(start,end,.19,.22)
    target=gaussian_filter1d(target,1.65,axis=0,mode='nearest')
    heading=gaussian_filter1d(heading,2.6,mode='nearest')
    target[:,1]=np.maximum(target[:,1],floor)
    return target,heading


def finish_body(frames,t,dense,rig):
    local,trans,world,positions={},{},{},{}
    dense_world,dense_local,dense_trans={},{},{}
    for i,name in enumerate(rig.names):
        p=rig.parent[i]
        parent=rig.names[p] if p>=0 else None
        delta=rig.point(name)-(rig.point(parent) if parent else 0)
        local[name]=Rotation.from_quat([q[name].as_quat() for q,_ in frames]) if name in frames[0][0] else Rotation.identity(len(t))
        trans[name]=np.array([tr.get(name,delta) for _,tr in frames])
        world[name]=world[parent]*local[name] if parent else local[name]
        positions[name]=positions[parent]+world[parent].apply(trans[name]) if parent else trans[name]
        if name in frames[0][1]:
            dense_trans[name]=curve(trans[name],t,dense,2.0 if name=='Hips' else 1.7)
        if name not in frames[0][0]:
            dense_world[name]=dense_world[parent] if parent else Rotation.identity(len(dense))
            continue
        secondary=name.startswith(('Tail','Skirt','Hair','Cowlick','Hand.'))
        width=2.45 if name in ('Head','Neck') else 1.8
        if secondary:
            dense_local[name]=orientation(local[name],t,dense,2.15)
            dense_world[name]=dense_world[parent]*dense_local[name]
        else:
            dense_world[name]=orientation(world[name],t,dense,width)
            dense_local[name]=dense_world[parent].inv()*dense_world[name] if parent else dense_world[name]

    hips=dense_trans['Hips']
    pelvis=dense_world['Hips']
    targets={}
    cap=np.full(len(dense),np.inf)
    for side in ('L','R'):
        thigh,shin,foot=(j+'.'+side for j in ('Thigh','Shin','Foot'))
        ankle=curve(positions[foot],t,dense,1.85)
        ankle[:,1]-=.012  # Bring the level sole to the floor after removing the source foot roll.
        direction=dense_world[foot].apply([0,0,1])
        heading=np.unwrap(np.arctan2(direction[:,0],direction[:,2]))
        ankle,heading=plant(ankle,heading,dense)
        offset=pelvis.apply(rig.point(thigh)-rig.point('Hips'))
        length=np.linalg.norm(rig.point(shin)-rig.point(thigh))+np.linalg.norm(rig.point(foot)-rig.point(shin))-.0004
        horizontal=np.linalg.norm((ankle-offset-hips)[:,[0,2]],axis=1)
        maximum=ankle[:,1]-offset[:,1]+np.sqrt(np.maximum(.003,length**2-horizontal**2))
        cap=np.minimum(cap,maximum)
        targets[side]=ankle,heading,offset
    # Settle the pelvis low enough for the planted shoe while easing into the bend.
    correction=np.minimum(0,cap-hips[:,1])
    hips[:,1]+=np.minimum(correction,gaussian_filter1d(correction,2.3,mode='nearest'))
    for side,(ankle,heading,offset) in targets.items():
        thigh,shin,foot=(j+'.'+side for j in ('Thigh','Shin','Foot'))
        upper=rig.point(shin)-rig.point(thigh)
        lower=rig.point(foot)-rig.point(shin)
        l1,l2=np.linalg.norm(upper),np.linalg.norm(lower)
        desired=pelvis.inv().apply(ankle-(hips+offset))
        axis=unit(desired)
        distance=np.clip(np.linalg.norm(desired,axis=1),abs(l1-l2)+.0004,l1+l2-.0003)
        # Preserve the filmed knee plane, with a forward pole as the knee straightens.
        pole=pelvis.inv().apply(dense_world[thigh].apply(upper))
        pole+=np.array([.019 if side=='L' else -.019,0,.085])
        pole-=axis*np.sum(pole*axis,axis=1)[:,None]
        pole=gaussian_filter1d(pole,1.7,axis=0)
        pole=unit(pole-axis*np.sum(pole*axis,axis=1)[:,None])
        along=(l1*l1-l2*l2+distance**2)/(2*distance)
        knee=axis*along[:,None]+pole*np.sqrt(np.maximum(0,l1*l1-along**2))[:,None]
        leg=shortest(upper,knee)
        calf=shortest(lower,axis*distance[:,None]-knee)
        dense_local[thigh]=leg
        dense_local[shin]=leg.inv()*calf
        dense_world[foot]=Rotation.from_euler('y',heading[:,None])
        dense_local[foot]=(pelvis*calf).inv()*dense_world[foot]
    # Palm anchors must come from the cleaned shoulder and elbow transforms.
    dense_pos={}
    for i,name in enumerate(rig.names):
        if i==0:
            dense_pos[name]=np.zeros((len(dense),3))
            continue
        parent=rig.names[rig.parent[i]]
        tr=dense_trans.get(name,rig.point(name)-rig.point(parent))
        dense_pos[name]=dense_pos[parent]+dense_world[parent].apply(tr)
    palms=np.stack([dense_pos['Hand.'+s]+dense_world['Hand.'+s].apply([k*.012,-.035,.016])
                    for s,k in [('L',1),('R',-1)]],axis=1)
    result=[({name:value[i] for name,value in dense_local.items()},
             {name:value[i].copy() for name,value in dense_trans.items()}) for i in range(len(dense))]
    return result,palms,hips


def finish_ball(raw,t,dense,impacts,radius):
    pos=curve(raw,t,dense,1.45)
    # Separate vertical arcs at impacts. Odd reflection preserves the incoming
    # and outgoing slopes, so smoothing never lifts a bounce off the floor.
    cuts=[0,*[int(np.argmin(abs(t-hit))) for hit in impacts],len(t)-1]
    for start,end in zip(cuts,cuts[1:]):
        y=raw[start:end+1,1].copy()
        if start: y[0]=radius
        if end<len(t)-1: y[-1]=radius
        pad=np.pad(y,6,mode='reflect',reflect_type='odd')
        y=gaussian_filter1d(pad,1.15)[6:-6]
        keep=(dense>=t[start]-1e-8)&(dense<=t[end]+1e-8)
        pos[keep,1]=np.maximum(radius,PchipInterpolator(t[start:end+1],y)(dense[keep]))
    visible=((dense>=10.39)&(dense<=24.04)).astype(float)
    return pos,visible
