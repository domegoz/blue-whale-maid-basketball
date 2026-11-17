"""Facial shape keys and secondary motion for the existing painted costume."""
import numpy as np
from scipy.interpolate import LinearNDInterpolator, PchipInterpolator
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation

from rig_animation import read_accessor, smooth


def spring(target,dt,frequency,damping):
    omega=2*np.pi*frequency
    out=np.empty_like(target)
    out[0]=target[0]
    v=0.0
    for i in range(1,len(out)):
        v+=(omega**2*(target[i]-out[i-1])-2*damping*omega*v)*dt
        out[i]=out[i-1]+v*dt
    return out


def face_shapes(p,uv):
    # The original front painting was projected with this mapping. Move its
    # existing artwork tangentially across the loft, keeping facial volume.
    x=403+300*p[:,0]
    y=695-300*p[:,1]
    u=(uv[:,0]-uv[:,0].min())/np.ptp(uv[:,0])
    front=np.abs(u-.5)<.232
    surface=LinearNDInterpolator(np.column_stack((x[front],y[front])),p[front,2])
    def tangent(dx,dy):
        result=np.zeros_like(p)
        use=front & (np.abs(dx)+np.abs(dy)>1e-5)
        depth=surface(x[use]+dx[use],y[use]+dy[use])
        result[use]=np.column_stack((dx[use]/300,-dy[use]/300,np.nan_to_num(depth-p[use,2])))
        return result
    center=np.where(x<402,340.,462.)
    margin=smooth(283,302,x)*(1-smooth(365,383,x))+smooth(413,431,x)*(1-smooth(498,515,x))
    # Collapse the painted eye to a shallow curved lash. Brow and cheek support
    # bands stretch over the opening and the iris retains a few percent height.
    line=331.0-3.4*((x-center)/34.)**2
    source=np.array([250,261,270,277,283,291,301,324,346,353,361,380,394.])
    closed=np.array([250,265,289,322.5,326.4,327.5,329.6,330.5,331.2,332,344,379,394.])
    final=PchipInterpolator(source,closed)(np.clip(y,source[0],source[-1]))
    curve=(line-331)*smooth(285,302,y)*(1-smooth(353,379,y))
    blink=tangent(np.zeros_like(x),(final-y+curve)*margin*((y>250)&(y<394)))
    iris=smooth(298,310,y)*(1-smooth(347,359,y))
    iris*=1-smooth(22,37,abs(x-center))
    gaze_x=tangent(5.0*iris,np.zeros_like(x))
    gaze_y=tangent(np.zeros_like(x),3.6*iris)
    eye=margin*smooth(265,278,y)*(1-smooth(353,367,y))
    upper=1-smooth(291,326,y)
    inner=np.clip((x-center)*np.where(x<402,1.,-1.)/30.,-1,1)
    cheek=np.exp(-((y-353)/15)**2)*margin
    focused=tangent(np.zeros_like(x),eye*upper*(4.0+6.3*inner)-.8*cheek)
    happy=tangent(np.zeros_like(x),-6.8*cheek+eye*upper*(1.8-3.8*inner))
    surprised=tangent(np.zeros_like(x),eye*(-6.6*upper+2.3*(1-upper)))
    brow_band=margin*np.exp(-((y-280.)/10.5)**2)
    brows=tangent(np.zeros_like(x),-8.5*brow_band*(.85+.32*inner))
    return [blink,gaze_x,gaze_y,focused,happy,surprised,blink*(x>402)[:,None],brows]


def add_details(frames,t,ball):
    fps=1/float(t[1]-t[0])
    dt=1/fps
    local_t=np.clip(t-8.8,0,40.5)
    hip=np.stack([tr['Hips'] for _,tr in frames])
    pelvis=Rotation.from_quat([q['Hips'].as_quat() for q,_ in frames])
    facing=pelvis.apply([0,0,1])
    yaw=np.unwrap(np.arctan2(facing[:,0],facing[:,2]))
    turn=gaussian_filter1d(np.gradient(yaw,dt),3,mode='nearest')
    speed=gaussian_filter1d(np.gradient(hip[:,0],dt),3)
    rise=gaussian_filter1d(np.gradient(hip[:,1],dt),3)
    shoulders=np.array([tr['Clavicle.L'][1]-tr['Clavicle.R'][1] for _,tr in frames])
    chest_pulse=gaussian_filter1d(np.gradient(shoulders,dt),2)
    trail=spring(np.tanh(turn/3.8),dt,2.2,.75)
    sideways=spring(np.tanh(speed/.92),dt,1.85,.70)
    lift=spring(np.clip(.67*abs(np.tanh(turn/3.8))+.48*abs(np.tanh(rise/1.2))+
                        .22*abs(np.tanh(speed/.8)),0,1),dt,2.3,.65)
    bow=spring(np.tanh(1.65*chest_pulse-.45*turn+.65*speed),dt,3.4,.58)
    bounce=spring(np.tanh(-1.3*rise+.42*np.gradient(speed,dt)),dt,2.7,.71)
    for i,(q,tr) in enumerate(frames):
        for name,factor in [('Skirt',.46),('SkirtHem',.54)]:
            follow=Rotation.from_euler('xyz',[1.8*bounce[i],-9.8*trail[i],2.6*sideways[i]],degrees=True)
            q[name]=q[name]*Rotation.from_rotvec(factor*follow.as_rotvec())
        tr['SkirtHem'][1]+=.019*lift[i]
    head=Rotation.from_quat([(q['Hips']*q['Spine']*q['Chest']*q['Neck']*q['Head']).as_quat() for q,_ in frames])
    direction=head.inv().apply(ball-hip-[0,.82,0])
    tracking=smooth(1.48,1.84,local_t)*(1-smooth(14.98,15.48,local_t))
    beats=np.array([0,.8,1.5,15.2,16.,17.4,18.65,19.95,21.3,23.8,25.05,
                    26.65,28.05,29.3,31.1,32.45,33.55,35.1,37.05,38.15,39.35,40.5])
    gx=PchipInterpolator(beats,[.04,-.32,-.65,-.52,.44,.18,-.38,.12,-.21,.63,
                              -.36,.25,-.46,.24,.11,-.58,-.32,.57,-.48,.31,.08,0])(local_t)
    gy=PchipInterpolator(beats,[.05,-.12,-.67,-.38,-.15,.34,.18,-.13,.24,.08,
                              .45,-.21,.20,-.28,-.22,.52,.31,-.23,-.15,.21,.08,0])(local_t)
    gx=gx*(1-tracking)+tracking*np.tanh(direction[:,0]*2.6)
    gy=gy*(1-tracking)+tracking*np.tanh(-direction[:,1]*1.4)
    blink=np.zeros(len(t))
    for center,rate in [(.99,.94),(4.16,1.05),(7.65,.92),(11.48,1.08),(15.68,.98),
                        (18.56,.94),(21.82,1.10),(25.48,.97),(28.90,1.02),
                        (32.56,.99),(36.20,1.1),(39.31,.97)]:
        blink+=smooth(center-.092*rate,center,local_t)*(1-smooth(center+.026*rate,center+.163*rate,local_t))
    eyes=np.column_stack((blink,gaussian_filter1d(gx,3.0)*(1-blink),
                          gaussian_filter1d(gy,3.5)*(1-blink)))
    # Focus, smile and surprise are authored on phrases in the film's timeline.
    cues=np.array([
        (0,.10,.40,0),(.8,.14,.60,0),(1.46,0,.13,.22),(1.93,0,.05,.94),
        (2.37,.16,.55,.04),(3.30,.28,.61,0),(4.18,.78,.07,0),(6.30,.50,.30,0),
        (8.0,.80,.16,0),(9.5,.30,.68,0),(10.7,.73,.08,0),(11.45,.06,.23,.75),
        (12.2,.48,.19,0),(13.03,.74,.03,0),(14.48,.14,.45,.16),(15.25,.02,.64,.22),
        (16.7,.30,.54,0),(18.35,.42,.35,0),(19.5,.53,.13,0),(20.12,.01,.18,.85),
        (20.61,.03,.86,0),(22.15,.12,.84,0),(23.70,.70,.14,0),
        (24.78,.78,.05,0),(25.08,0,.12,.98),(25.63,.05,.94,.05),(27.0,.13,.84,0),
        (28.4,.53,.37,0),(29.36,.12,.90,0),(30.40,.23,.68,0),(31.94,.56,.28,0),
        (32.34,.08,.19,.72),(33.06,.31,.54,0),(34.34,.08,.70,.08),
        (35.64,.21,.62,0),(36.9,.04,.86,0),(38.37,.02,.85,.08),(39.14,.12,.62,0),
        (40.08,0,.97,0),(40.5,.03,.89,0)])
    mood=PchipInterpolator(cues[:,0],cues[:,1:],axis=0)(local_t)
    mood/=np.maximum(1,mood.sum(1,keepdims=True))
    eye_mood=mood*(1-blink[:,None])**1.3
    wink=smooth(29.40,29.53,local_t)*(1-smooth(29.63,29.81,local_t))
    eyes[:,1:]*=(1-.8*wink[:,None])
    eye_mood*=1-.9*wink[:,None]
    from opening_acting import facial_performance
    eyes,mood=facial_performance(t,np.column_stack((eyes,eye_mood,wink)),mood)
    return dict(eye=eyes,mood=mood,lift=lift,trail=trail,
                side=sideways,bow=bow,bounce=bounce)


def add_morphs(doc,buf,acc,timeacc,nodes,channels,samplers,control):
    from garment_dynamics import skirt_offsets
    from mouth_mesh import make_mouth
    mouth=make_mouth(doc,buf,acc,nodes)
    skirt=('Midnight embroidered','Scalloped ivory','Skirt underside','Skirt ribbon','Whale apron')
    for mesh in doc['meshes']:
        name=mesh['name']
        if name=='Crown and flowing back hair':
            attr=mesh['primitives'][0]['attributes']
            shell=read_accessor(doc,buf,attr['POSITION'])
            arc=read_accessor(doc,buf,attr['TEXCOORD_0'])[:,1]
            pole=arc==arc.min()
            shell[pole]=shell[pole].mean(axis=0)
            attr['POSITION']=acc(shell,'VEC3',True)
            continue
        if name=='Hair lining':
            # Separate the lining from the outer cap at the crown, exposed in the bow.
            attr=mesh['primitives'][0]['attributes']
            shell=read_accessor(doc,buf,attr['POSITION'])
            top=smooth(1.62,1.99,shell[:,1])
            shell[:,0]*=1-.035*top
            shell[:,2]-=(shell[:,2]+.062)*.035*top
            shell[:,1]-=.034*top
            attr['POSITION']=acc(shell,'VEC3',True)
            continue
        if name not in control['front_hair'] and not name.startswith(('Face ·','Sailor throat bow','Mouth expression',*skirt)):
            continue
        prim=mesh['primitives'][0]
        p=read_accessor(doc,buf,prim['attributes']['POSITION'])
        x,y,z=p.T
        if name.startswith('Face ·'):
            keys=['Blink','Gaze across','Gaze down','Determined eyes','Smile eyes','Eyes wide','Wink','Raise brows']
            offsets=face_shapes(p,read_accessor(doc,buf,prim['attributes']['TEXCOORD_0']))
            weights=control['eye']
        elif name=='Mouth expression':
            offsets=mouth
            keys=['Press lips','Smile','Round mouth','Open vowel','Spread vowel','Pursed vowel','Closed consonant','Lift mouth corners']
            weights=control['mood']
        elif name in control['front_hair']:
            p,offsets,weights=control['front_hair'][name]
            keys=[f'Strand bend {i+1}' for i in range(len(offsets))]
            attr=prim['attributes']
            attr['POSITION']=acc(p,'VEC3',True)
            joints=np.zeros((len(p),4),'u1')
            joints[:,0]=doc['skins'][0]['joints'].index(nodes['Head'])
            bind=np.zeros((len(p),4),'f4')
            bind[:,0]=1
            if name=='Crown cowlick':
                arc=read_accessor(doc,buf,attr['TEXCOORD_0'])[:,1]
                soft=smooth(.04,.80,(arc-arc.min())/np.ptp(arc))
                joints[:,1]=doc['skins'][0]['joints'].index(nodes['Cowlick'])
                bind[:,0],bind[:,1]=1-soft,soft
            attr['JOINTS_0']=acc(joints,'VEC4',component=5121)
            attr['WEIGHTS_0']=acc(bind,'VEC4')
        elif name.startswith('Sailor throat bow'):
            flap=smooth(.022,.096,abs(x))+.38*(1-smooth(.86,.936,y))
            zeros=np.zeros_like(x)
            offsets=[np.column_stack((zeros,.022*flap*np.sign(x),.025*flap*np.sign(x))),
                     np.column_stack((zeros,.013*flap,.021*flap))]
            keys=['Ribbon flutter','Ribbon lift']
            weights=np.column_stack((control['bow'],control['bounce']))
        else:
            offsets=skirt_offsets(p,control['skirt_modes'])
            keys=[f'Cloth mode {i+1}' for i in range(len(offsets))]
            weights=control['skirt_weights']
            if name.startswith(('Whale apron','Skirt ribbon')):
                free=(1-smooth(.37,.70,y)) if name.startswith('Whale apron') else (1-smooth(.27,.40,y))
                offsets.append(np.column_stack((.022*free,.008*free,.023*free)))
                keys.append('Fabric sway')
                weights=np.column_stack((weights,.6*control['side']+.4*control['bow']))
        prim['targets']=[{'POSITION':acc(delta,'VEC3',True)} for delta in offsets]
        if name.startswith(skirt) or name in control['front_hair']:
            faces=read_accessor(doc,buf,prim['indices']).reshape(-1,3)
            def normals(vertices):
                a,b,c=vertices[faces].transpose(1,0,2)
                area=np.cross(b-a,c-a)
                n=np.zeros_like(vertices)
                for k in range(3): np.add.at(n,faces[:,k],area)
                return n/np.maximum(1e-10,np.linalg.norm(n,axis=1,keepdims=True))
            for target,delta in zip(prim['targets'],offsets):
                dn=normals(p+.5*delta)-normals(p-.5*delta)
                target['NORMAL']=acc(dn,'VEC3')
        mesh['weights']=[0.]*len(keys)
        mesh.setdefault('extras',{})['targetNames']=keys
        output=acc(weights.ravel(),'SCALAR')
        channels.append(dict(sampler=len(samplers),target=dict(node=nodes[name],path='weights')))
        samplers.append(dict(input=timeacc,output=output,interpolation='LINEAR'))
