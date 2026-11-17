"""Bake world-space constrained hair chains and a pinned, two-sided skirt cage."""
import numpy as np
from numba import njit
from scipy.ndimage import gaussian_filter1d,maximum_filter1d
from scipy.spatial.transform import Rotation

from rig_animation import smooth


def poses(frames,rig):
    position,rotation={},{}
    for i,name in enumerate(rig.names):
        p=rig.parent[i]
        parent=rig.names[p] if p>=0 else None
        delta=rig.point(name)-(rig.point(parent) if parent else 0)
        local=np.array([tr.get(name,delta) for _,tr in frames])
        quat=Rotation.from_quat([q.get(name,Rotation.identity()).as_quat() for q,_ in frames])
        if parent:
            rotation[name]=rotation[parent]*quat
            position[name]=position[parent]+rotation[parent].apply(local)
        else:
            rotation[name],position[name]=quat,local
    return position,rotation


@njit(cache=True)
def solve(goal,edges,strength,stiffness,limits,floors,colliders,radii,dt):
    count,n,_=goal.shape
    output=np.empty_like(goal)
    p=goal[0].copy()
    velocity=np.zeros_like(p)
    substeps=4
    h=dt/substeps
    for fr in range(count):
        prev=max(fr-1,0)
        for sub in range(substeps):
            u=(sub+1)/substeps
            rest=goal[prev]*(1-u)+goal[fr]*u
            sticks=colliders[prev]*(1-u)+colliders[fr]*u
            start=p.copy()
            for j in range(n):
                k=stiffness[j]
                if k<=0:
                    p[j]=rest[j]
                    continue
                for a in range(3):
                    parent_v=(goal[fr,j,a]-goal[prev,j,a])/dt
                    acc=k*(rest[j,a]-p[j,a])-1.05*np.sqrt(k)*(velocity[j,a]-.18*parent_v)
                    if a==1: acc-=1.6
                    velocity[j,a]+=h*acc
                    p[j,a]+=h*velocity[j,a]
            for iteration in range(4):
                for e in range(len(edges)):
                    a,b=edges[e]
                    d=p[b]-p[a]
                    mag=np.sqrt(d@d)+1e-10
                    r=rest[b]-rest[a]
                    corr=d*((mag-np.sqrt(r@r))/mag)*strength[e]
                    wa=1.0 if stiffness[a]>0 else 0.
                    wb=1.0 if stiffness[b]>0 else 0.
                    if wa+wb>0:
                        p[a]+=corr*wa/(wa+wb)
                        p[b]-=corr*wb/(wa+wb)
                for j in range(n):
                    if stiffness[j]<=0: continue
                    d=p[j]-rest[j]
                    mag=np.sqrt(d@d)+1e-10
                    if mag>limits[j]:
                        p[j]=rest[j]+d*(limits[j]/mag)
                    for c in range(len(radii)):
                        a=sticks[c,0]; ab=sticks[c,1]-a
                        v=p[j]-a
                        along=max(0.,min(1.,(v@ab)/(ab@ab+1e-10)))
                        direction=v-ab*along
                        dist=np.sqrt(direction@direction)+1e-10
                        if dist<radii[c]:
                            p[j]+=direction*((radii[c]-dist)/dist)
                    p[j,1]=max(p[j,1],floors[j])
            velocity=(p-start)/h
        output[fr]=p
    return output


def swing(rest,directions):
    a=rest/np.linalg.norm(rest)
    b=directions/np.linalg.norm(directions,axis=1,keepdims=True)
    return Rotation.from_quat(np.column_stack((np.cross(a,b),1+b@a)))


ROWS=np.array([.716,.575,.405,.202])
SECTORS=16


def skirt_offsets(p,modes):
    x,y,z=p.T
    row=np.interp(y,ROWS[::-1],np.arange(4)[::-1])
    # Register every garment layer to the same panel angle and sewn height.
    angle=np.mod(np.arctan2(x/.51,(z-.15)/.419),2*np.pi)*SECTORS/(2*np.pi)
    lo=row.astype(int);hi=np.minimum(lo+1,3);u=(row-lo)[:,None]
    a=angle.astype(int)%SECTORS;b=(a+1)%SECTORS;v=(angle%1)[:,None]
    hem_extension=1+.12*np.clip((ROWS[-1]-y)/.075,0,1)
    result=[]
    for m in modes:
        ring0=m[lo,a]*(1-v)+m[lo,b]*v
        ring1=m[hi,a]*(1-v)+m[hi,b]*v
        result.append(((1-u)*ring0+u*ring1)*hem_extension[:,None])
    return result


@njit(cache=True)
def exits(vertices,center,rays,radius):
    out=np.zeros(len(rays))
    for j in range(len(vertices)):
        x,y,z=vertices[j]-center
        ds=x*x+y*y+z*z
        if ds>1.7: continue
        for k in range(len(rays)):
            ray=rays[k]
            along=x*ray[0]+y*ray[1]+z*ray[2]
            perp=ds-along*along
            if perp<radius*radius:
                out[k]=max(out[k],along+np.sqrt(max(0.,radius*radius-perp)))
    return out


def route_ball(ball,visible,hip,hip_position,rest_hips,vertices,palms,coeff=None,offsets=None):
    # Follow the existing crossover around the actual dress outline. A sphere
    # cannot move through the bell of the skirt just because it fits the source shot.
    center=hip_position+hip.apply([0,-.20,.015])
    angles=np.arange(16)*2*np.pi/16
    directions=np.column_stack((np.sin(angles),np.zeros(16),np.cos(angles)))
    active=np.flatnonzero(visible)
    choices=np.empty((len(active),16,3))
    costs=np.zeros((len(active),16))
    parent=np.zeros((len(active),16),int)
    rotation=hip.as_matrix()
    def geometry(i):
        p=vertices if offsets is None else vertices+np.einsum('i,ijk->jk',coeff[i],offsets)
        local=(ball[i]-hip_position[i])@rotation[i]+rest_hips
        return p,local
    for k,i in enumerate(active):
        points,local=geometry(i)
        distances=exits(points[::2],local,directions@rotation[i],.234)
        offset=distances[:,None]*directions
        choices[k]=offset
        palm=palms[i]
        old=np.linalg.norm(palm-ball[i],axis=1)
        grip=1-smooth(.245,.318,old)
        new_dist=np.linalg.norm(ball[i]+offset[:,None,:]-palm[None,:,:],axis=-1)
        costs[k]=distances**2+1.8*offset[:,0]**2+5*np.sum(grip*(new_dist-.228)**2,axis=1)
        if k:
            change=choices[k-1][:,None,:]-offset[None,:,:]
            transfer=costs[k-1][:,None]+14*np.sum(change*change,axis=-1)
            parent[k]=np.argmin(transfer,axis=0)
            costs[k]+=transfer[parent[k],np.arange(16)]
    best=np.empty(len(active),int)
    best[-1]=np.argmin(costs[-1])
    for k in range(len(active)-1,0,-1): best[k-1]=parent[k,best[k]]
    offsets_world=choices[np.arange(len(active)),best]
    direction=gaussian_filter1d(offsets_world,2.1,axis=0)
    mag=np.linalg.norm(direction,axis=1)
    fallback=ball[active]-center[active];fallback[:,1]=0
    fallback/=np.maximum(.001,np.linalg.norm(fallback,axis=1,keepdims=True))
    direction=np.where(mag[:,None]>.0001,direction/np.maximum(.0001,mag[:,None]),fallback)
    needed=np.zeros(len(ball))
    for k,i in enumerate(active):
        points,local=geometry(i)
        needed[i]=exits(points,local,direction[k:k+1]@rotation[i],.233)[0]
    pad=gaussian_filter1d(maximum_filter1d(needed,7),2.0)
    distance=np.maximum(needed,pad)
    ball[active]+=direction*distance[active,None]
    return float(needed.max())


def keep_grip(frames,pos,rot,palms,original,ball,visible,pass_weight):
    original_grip=1-smooth(.247,.324,np.linalg.norm(original[:,None]-palms,axis=2).min(1))
    shoulder=np.stack([pos['UpperArm.L'],pos['UpperArm.R']],axis=1)
    reachable=np.linalg.norm(ball[:,None]-shoulder,axis=2)
    for side,name in enumerate(('L','R')):
        upper,fore,hand,clav=(key+'.'+name for key in ('UpperArm','Forearm','Hand','Clavicle'))
        palm=palms[:,side]
        old=np.linalg.norm(original-palm,axis=1)
        gap=np.linalg.norm(ball-palm,axis=1)
        amount=(1-smooth(.247,.324,old))*visible
        # Both hands may receive the ball during the partly hidden waist pass.
        catch=1-smooth(-.07,.025,reachable[:,side]-reachable[:,1-side])
        amount=np.maximum(amount,original_grip*catch*pass_weight)
        displacement=(ball-palm)*(amount*(gap-.225)/np.maximum(.01,gap))[:,None]
        displacement=gaussian_filter1d(displacement,1.7,axis=0)
        a=pos[upper].copy();b=pos[fore];c=pos[hand];target=c+displacement
        upper_len=np.linalg.norm(b-a,axis=1);lower_len=np.linalg.norm(c-b,axis=1)
        def unit(v): return v/np.maximum(1e-8,np.linalg.norm(v,axis=1,keepdims=True))
        axis=unit(target-a)
        dist=np.linalg.norm(target-a,axis=1)
        lift=np.minimum(.058,np.maximum(0,dist-upper_len-lower_len+.009))
        shift=gaussian_filter1d(axis*lift[:,None],1.3,axis=0)
        a+=shift
        axis=unit(target-a)
        reach=np.clip(np.linalg.norm(target-a,axis=1),np.abs(upper_len-lower_len)+.0002,upper_len+lower_len-.0002)
        pole=b-a-np.sum((b-a)*axis,axis=1)[:,None]*axis
        along=(upper_len**2-lower_len**2+reach**2)/(2*reach)
        elbow=axis*along[:,None]+unit(pole)*np.sqrt(np.maximum(0,upper_len**2-along**2))[:,None]
        def redirect(old,new):
            u,v=unit(old),unit(new)
            return Rotation.from_quat(np.column_stack((np.cross(u,v),1+np.sum(u*v,axis=1))))
        u=redirect(b-pos[upper],elbow)*rot[upper]
        l=redirect(c-b,axis*reach[:,None]-elbow)*rot[fore]
        local_u=rot[clav].inv()*u;local_f=u.inv()*l;local_h=l.inv()*rot[hand]
        delta=rot['Chest'].inv().apply(shift)
        for k in np.flatnonzero(np.linalg.norm(displacement,axis=1)>.0005):
            q,tr=frames[k];q[upper]=local_u[k];q[fore]=local_f[k];q[hand]=local_h[k]
            tr[clav]=tr[clav]+delta[k]


def simulate(frames,t,ball,visible,rig,cloth_meshes):
    pos,rot=poses(frames,rig)
    dt=float(t[1]-t[0])
    hip,head=rot['Hips'],rot['Head']
    cloth_vertices=np.vstack(cloth_meshes)
    palms=np.stack([pos['Hand.'+s]+rot['Hand.'+s].apply([k*.012,-.035,.016])
                    for s,k in [('L',1),('R',-1)]],axis=1)
    original_ball=ball.copy()
    # The reference briefly hides the ball behind the performer's waist. Carry
    # it at the maid's hip here, above the widest and least flexible part of her dress.
    phase=t-8.8
    waist_pass=smooth(2.55,2.94,phase)*(1-smooth(3.40,3.84,phase))
    ball[:,1]+=waist_pass*np.maximum(0,pos['Hips'][:,1]-.036-ball[:,1])
    preclear=route_ball(ball,visible,hip,pos['Hips'],rig.point('Hips'),cloth_vertices,palms)
    def capsule(a,b):
        return np.stack((pos[a],pos[b]),axis=1)
    hidden_ball=ball.copy()
    hidden_ball[visible<.5]=[0,8,0]
    body=[capsule('Thigh.L','Shin.L'),capsule('Thigh.R','Shin.R'),
          capsule('Shin.L','Foot.L'),capsule('Shin.R','Foot.R'),
          capsule('Hips','Neck'),np.repeat(hidden_ball[:,None,:],2,axis=1)]
    collisions=np.stack(body,axis=1)
    radii=np.array([.112,.112,.079,.079,.174,.243])
    def world(rel,r,origin):
        return np.einsum('tij,tnj->tni',r.as_matrix(),rel)+origin[:,None,:]
    head_rest=rig.point('Head')
    samples=[];edges=[];weights=[]
    for i,side in enumerate(('R','C','L')):
        root=rig.point('HairUpper.'+side)
        mid=rig.point('HairTip.'+side)
        tip=mid+np.array([(-1+i)*.037,-.445,-.024])
        samples.extend([root,mid,tip])
        edges.extend([(3*i,3*i+1),(3*i+1,3*i+2),(3*i,3*i+2)])
        weights.extend([.95,.92,.14])
    samples=np.array(samples)
    hair_rest=np.broadcast_to(samples-head_rest,(len(t),9,3))
    hair_goal=world(hair_rest,head,pos['Head'])
    hair=solve(hair_goal,np.array(edges),np.array(weights),np.array([0.,140.,65.,0.,158.,71.,0.,135.,62.]),
               np.tile([0,.15,.28],3),np.tile([0,.05,.045],3),collisions,radii,dt)
    for i,side in enumerate(('R','C','L')):
        start=3*i
        bone=swing(samples[start+1]-samples[start],head.inv().apply(hair[:,start+1]-hair[:,start]))
        tip=swing(samples[start+2]-samples[start+1],head.inv().apply(hair[:,start+2]-hair[:,start+1]))
        local=bone.inv()*tip
        for k,(q,_) in enumerate(frames):
            q['HairUpper.'+side]=bone[k]
            q['HairTip.'+side]=local[k]

    theta=np.arange(SECTORS)*2*np.pi/SECTORS
    rx=np.array([.182,.303,.433,.508]);rz=np.array([.154,.251,.351,.419])
    rings=np.stack(np.broadcast_arrays(rx[:,None]*np.sin(theta),ROWS[:,None],
                                      .15+rz[:,None]*np.cos(theta)),axis=-1)
    goals=np.broadcast_to(rings,(len(t),*rings.shape)).copy()
    forward=hip.apply([0,0,1])
    yaw=np.unwrap(np.arctan2(forward[:,0],forward[:,2]))
    omega=gaussian_filter1d(np.gradient(yaw,dt),2.4)
    spread=(1-np.exp(-np.square(omega/4.3)))[:,None]
    amount=np.array([0,.012,.037,.075])[None,:]*spread
    goals[:,:,:,0]+=amount[:,:,None]*np.sin(theta)
    goals[:,:,:,2]+=amount[:,:,None]*np.cos(theta)
    compression=np.clip(.66-pos['Hips'][:,1],0,.24)
    goals[:,:,:,1]+=(compression[:,None]*[0,.13,.38,.72]+amount*.65)[:,:,None]
    edges=[];weights=[]
    for row in range(4):
        for k in range(SECTORS):
            cur=row*SECTORS+k
            def link(r,sector,w):
                edges.append((cur,r*SECTORS+sector%SECTORS));weights.append(w)
            if row:
                link(row,k+1,.62)
                link(row,k+2,.15)
                link(row-1,k,.9)
                link(row-1,k-1,.23)
                link(row-1,k+1,.23)
            if row>1: link(row-2,k,.18)
    fixed=np.repeat([0.,184.,106.,66.],SECTORS)
    goals=world(goals.reshape(len(t),-1,3)-rig.point('Hips'),hip,pos['Hips'])
    simulated=solve(goals,np.array(edges),np.array(weights),fixed,np.repeat([0,.075,.146,.23],SECTORS),
                    np.repeat([0,.12,.105,.090],SECTORS),collisions,radii,dt)
    local=np.einsum('tij,tnj->tni',hip.inv().as_matrix(),simulated-pos['Hips'][:,None])+rig.point('Hips')
    deltas=local.reshape(len(t),4,SECTORS,3)-rings
    # A shared spatial basis keeps seams continuous and plays in standard glTF viewers.
    flat=deltas.reshape(len(t),-1)
    _,_,v=np.linalg.svd(flat,full_matrices=False)
    basis=v[:7]
    coeff=flat@basis.T
    amp=np.maximum(np.abs(coeff).max(0),1e-7)
    modes=(basis*amp[:,None]).reshape(-1,4,SECTORS,3)
    captured=1-np.sum((flat-coeff@basis)**2)/np.sum(flat**2)
    print(f'Skirt cage 64 particles, retained motion energy {captured:.2%}; peak hair lag {np.max(np.linalg.norm(hair-hair_goal,axis=2)):.3f} m',flush=True)
    final_clear=route_ball(ball,visible,hip,pos['Hips'],rig.point('Hips'),cloth_vertices,palms,
                           coeff/amp,np.array(skirt_offsets(cloth_vertices,modes)))
    print(f'Ball routing around costume: {preclear:.3f} m initial / {final_clear:.3f} m final',flush=True)
    keep_grip(frames,pos,rot,palms,original_ball,ball,visible,waist_pass)
    for q,tr in frames:
        for name,parent in [('Skirt','Hips'),('SkirtHem','Skirt')]:
            q[name]=Rotation.identity()
            tr[name]=rig.point(name)-rig.point(parent)
    return dict(skirt_modes=modes,skirt_weights=coeff/amp,head_pose=(pos['Head'],head))
