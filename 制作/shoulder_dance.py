"""Four-second basketball shoulder-step test, in glTF Y-up meters."""
from pathlib import Path
import io,json,struct,sys,subprocess
import numpy as np
from PIL import Image
from scipy.interpolate import PchipInterpolator
from scipy.spatial.transform import Rotation
from rig_animation import ROOT,OUTPUT,load_glb,read_accessor,skeleton,rot,align,smooth

OUT=OUTPUT.parent/'篮球舞_4秒验证'
OUT.mkdir(exist_ok=True)
GLB=OUT/'蓝鲸女仆_篮球舞.glb'
MP4=OUT/'篮球舞_4秒.mp4'
R=skeleton()
DURATION=4.;FPS=30;RAD=.124
def channel(keys):
    return PchipInterpolator(*np.array(keys,dtype=float).T)
yaw=channel([(0,0),(.35,-12),(.62,-6),(1.04,39),(1.32,68),(1.60,81),(1.91,61),(2.24,83),(2.50,60),(2.78,77),(3.13,37),(3.58,-9),(3.82,-2),(4,0)])
shift=channel([(0,0),(.20,-.035),(.40,-.085),(.72,.073),(1.03,.122),(1.35,.026),(1.57,.114),(1.89,-.056),(2.21,.139),(2.53,-.055),(2.77,.105),(3.06,.014),(3.36,-.100),(3.72,.016),(4,0)])
drop=channel([(0,.027),(.20,.037),(.37,.122),(.69,.048),(.98,.088),(1.20,.040),(1.40,.117),(1.58,.049),(1.80,.095),(1.96,.145),(2.22,.047),(2.43,.093),(2.58,.131),(2.78,.047),(3.00,.10),(3.25,.051),(3.46,.107),(3.75,.047),(4,.027)])
accent=channel([(0,0),(.45,0),(.82,-.14),(1.17,.15),(1.40,-.77),(1.57,1),(1.72,.50),(1.96,-.79),(2.19,1),(2.34,.34),(2.57,-.66),(2.75,.90),(2.91,.14),(3.11,-.22),(3.45,.16),(3.70,0),(4,0)])
turn=channel([(0,0),(.63,0),(1.22,.89),(1.55,1),(2.75,1),(3.15,.55),(3.64,0),(4,0)])
press=channel([(0,0),(.14,.43),(.34,1),(.53,.31),(.72,0),(4,0)])
ct=channel([(0,0),(.39,-.55),(.72,.15),(1.19,.6),(1.46,.82),(2.77,.86),(3.35,.36),(3.77,0),(4,0)])
def val(fn,t):return float(fn(np.clip(t,0,4)))
# Each swing connects two planted world positions. Flat soles remain on the floor between swings.
steps={
 'L':[(.15,.61,(.245,.095,12),.095),(1.10,1.43,(.178,-.062,69),.103),(2.97,3.32,(.203,.09,16),.095),(3.66,3.98,(.137,.144,0),.058)],
 'R':[(.64,1.10,(-.161,.355,50),.135),(2.55,2.97,(-.228,.278,37),.117),(3.32,3.66,(-.137,.144,0),.090)]}
def foot_target(side,t):
    x,z,angle=((-1 if side=='R' else 1)*.137,.144,0)
    for start,end,dst,height in steps[side]:
        if t<start:break
        if t>=end:x,z,angle=dst;continue
        f=(t-start)/(end-start);a=float(smooth(0,1,f));lift=height*np.sin(np.pi*f)**2
        return np.array([x+(dst[0]-x)*a,.091+lift,z+(dst[1]-z)*a]),rot(y=angle+(dst[2]-angle)*a)*rot(x=8*np.sin(2*np.pi*f)*np.sin(np.pi*f))
    return np.array([x,.091,z]),rot(y=angle)
def unit(v):return v/np.linalg.norm(v)
def ik(side,hip_at,hip_rot,t):
    suf='.'+side
    thigh,knee,ankle=[R.point(a+suf) for a in ['Thigh','Shin','Foot']]
    world_ankle,world_footrot=foot_target(side,t)
    start=hip_at+hip_rot.apply(thigh-R.point('Hips'))
    disp=hip_rot.inv().apply(world_ankle-start);axis=unit(disp);dist=np.linalg.norm(disp)
    upper=knee-thigh;lower=ankle-knee
    l1,l2=np.linalg.norm(upper),np.linalg.norm(lower)
    if not abs(l1-l2)<dist<l1+l2+1e-7:raise ValueError(('unreachable foot',t,side,dist,l1+l2))
    pole=np.array([-.15 if side=='R' else .15,0,1]);bend=unit(pole-axis*np.dot(axis,pole))
    h=(l1*l1-l2*l2+dist*dist)/(2*dist)
    mid=axis*h+bend*np.sqrt(max(0,l1*l1-h*h))
    u=align(upper,mid);l=align(lower,disp-mid)
    return {'Thigh'+suf:u,'Shin'+suf:u.inv()*l,'Foot'+suf:(hip_rot*l).inv()*world_footrot}
def pose(t,radius=RAD):
    a=val(accent,t);delayed=val(accent,t-.105);post=val(accent,t-.18);k=val(turn,t)
    dr=val(press,t);roll=val(shift,t);bend=val(drop,t);cr=val(ct,t);ang=val(yaw,t)+7*k
    hiprot=rot(y=ang)*rot(x=3*k+dr*3,z=-2.4*a)
    hips=R.point('Hips')+np.array([roll,-bend,.026*k-.025*dr])
    for side in ('R','L'):
        thigh,knee,ankle=[R.point(b+'.'+side) for b in ['Thigh','Shin','Foot']]
        offset=hiprot.apply(thigh-R.point('Hips'));foot=foot_target(side,t)[0]
        diff=(hips+offset-foot)[[0,2]]
        length=np.linalg.norm(knee-thigh)+np.linalg.norm(ankle-knee)-.0015
        hips[1]=min(hips[1],foot[1]+np.sqrt(max(.001,length**2-diff@diff))-offset[1])
    q={'Hips':hiprot,'Spine':rot(x=6*k+3*a+6*dr,y=6*k+5*a,z=-4*a-2*dr),
       'Chest':rot(x=8*k+5*a+3*dr,y=11*k+17*a,z=-7*a-3*dr),
       'Neck':rot(x=-3*k-2*a,y=-8*k-3*a,z=3.5*a),
       'Head':rot(x=-7*k-3.3*delayed-2*dr,y=-36*k-9*delayed,z=3*delayed+2*dr),
       'Skirt':rot(x=-2.9*k-3*delayed,y=-3.9*delayed,z=2.6*delayed),
       'SkirtHem':rot(x=-2*post,y=-5.3*post,z=3.5*post),
       'Cowlick':rot(x=5*(a-post),z=-8*(a-post),y=-4*post),
       'TailBase':rot(x=4*k,y=-7*delayed,z=3*a),
       'TailMid':rot(x=3*a,y=-6*post,z=4*delayed),
       'TailFlukes':rot(x=5*post,y=-5*post,z=-3*(a-post))}
    lift=float(smooth(.021,.128,bend))
    tr={'Hips':hips,'Skirt':R.point('Skirt')-R.point('Hips')+np.array([0,.018*lift,0]),
        'SkirtHem':R.point('SkirtHem')-R.point('Skirt')+np.array([0,.063*lift,0])}
    for side in ('R','L'):
        sign=-1 if side=='R' else 1
        suf='.'+side
        q.update(ik(side,hips,hiprot,t))
        shoulder_pop=(1 if side=='R' else -.4)*a*k
        q['Clavicle'+suf]=rot(x=4*shoulder_pop,y=sign*(-5*shoulder_pop),z=sign*(9*shoulder_pop+2*k))
        anchor=R.point('Clavicle'+suf)-R.point('Chest')
        tr['Clavicle'+suf]=anchor+np.array([sign*.017*shoulder_pop,.049*shoulder_pop,.054*shoulder_pop])
        if side=='R':
            up=np.array([-.127-.038*k,-.100-.018*k-.052*dr,.114-.144*k])
            fore=np.array([-.103+.185*k+.019*a,-.067+.069*k-.105*dr,.154+.009*a])
        else:
            up=np.array([.140+.038*(1-k)+.025*dr,-.137+.087*k+.034*a,.042+.039*k])
            fore=np.array([.092-.235*k-.043*a,-.136+.235*k+.022*a,.077+.042*k])
        upper=align(R.point('Forearm'+suf)-R.point('UpperArm'+suf),up)
        forearm=align(R.point('Hand'+suf)-R.point('Forearm'+suf),fore)
        q['UpperArm'+suf]=upper;q['Forearm'+suf]=upper.inv()*forearm
        q['Hand'+suf]=rot(x=-7*k-12*dr if side=='R' else 5*a,y=sign*8*k,z=sign*7*a)
    for side,sign in [('L',1),('C',0),('R',-1)]:
        q['HairUpper.'+side]=rot(x=-4*k-3*a,y=-6*k-6*(a-delayed),z=3.4*a+sign*1.4*dr)
        q['HairTip.'+side]=rot(x=-2*delayed,y=-5*delayed-5*(delayed-post),z=4*delayed)
    # Evaluate the chain to the palm so the ball is held in the same place after the catch.
    wt=[];wq=[]
    for i,name in enumerate(R.names):
        par=R.parent[i]
        local=tr.get(name,R.point(name)-(R.point(R.names[par]) if par>=0 else 0))
        qr=q.get(name,Rotation.identity())
        if par>=0:wt.append(wt[par]+wq[par].apply(local));wq.append(wq[par]*qr)
        else:wt.append(local);wq.append(qr)
    hand=R.index['Hand.R'];chest=R.index['Chest']
    palm=wt[hand]+wq[hand].apply([-.018,-.033,.008])
    contact=np.array([-.043,-.071,.091])
    held=palm+wq[chest].apply(contact+(radius-RAD)*unit(contact))
    return q,tr,held,wq[chest]
RELEASE=.05;IMPACT=.35;CATCH=.74
def ball(t,held,radius,start,end):
    if t<RELEASE or t>CATCH:return held
    s=(t-RELEASE)/(CATCH-RELEASE);p=start*(1-s)+end*s
    p[0]-=.022*np.sin(np.pi*s);g=6.3
    if t<=IMPACT:
        d=IMPACT-RELEASE;u=t-RELEASE;v=(radius-start[1]+.5*g*d*d)/d;p[1]=start[1]+v*u-.5*g*u*u
    else:
        d=CATCH-IMPACT;u=t-IMPACT;v=(end[1]-radius+.5*g*d*d)/d;p[1]=radius+v*u-.5*g*u*u
    return p

def ball_mesh():
    u,v=np.meshgrid(np.linspace(0,1,65),np.linspace(0,1,43));a=u*2*np.pi;h=(.5-v)*np.pi
    n=np.stack([np.cos(h)*np.sin(a),np.sin(h),np.cos(h)*np.cos(a)],axis=-1).reshape(-1,3)
    i=np.arange(65*42).reshape(42,65)[:,:-1].ravel()
    f=np.vstack([np.c_[i,i+65,i+1],np.c_[i+1,i+65,i+66]])
    f=f[np.linalg.norm(np.cross(n[f[:,1]]-n[f[:,0]],n[f[:,2]]-n[f[:,0]]),axis=1)>1e-10]
    return n,np.stack([u,v],axis=-1).reshape(-1,2),f
def ball_texture():
    u,v=np.meshgrid(np.linspace(0,1,512),np.linspace(0,1,256));a=u*2*np.pi;h=(.5-v)*np.pi
    x=np.cos(h)*np.sin(a);y=np.sin(h);z=np.cos(h)*np.cos(a)
    seam=np.minimum.reduce([abs(y),abs(x),abs(abs(z)-(.55+.22*y*y))])
    ink=1-smooth(.012,.034,seam)
    noise=3*np.sin(2*np.pi*(u*125+v*27))*np.cos(2*np.pi*(v*95-u*7))+2*np.cos(2*np.pi*(u*131-v*63))
    orange=np.array([223,125,59])+noise[...,None]+6*(y*.5+.4*z)[...,None]
    return Image.fromarray(np.clip(orange*(1-ink[...,None])+np.array([68,49,44])*ink[...,None],0,255).astype('uint8'))

def write(destination=GLB,radius=RAD):
    start,end=pose(RELEASE,radius)[2],pose(CATCH,radius)[2]
    j,b=load_glb(OUTPUT);skin=j['skins'][0];nodes=j['nodes']
    def rawview(data,target=None):
        while len(b)%4:b.append(0)
        view={'buffer':0,'byteOffset':len(b),'byteLength':len(data)}
        if target:view['target']=target
        i=len(j['bufferViews']);j['bufferViews'].append(view);b.extend(data);return i
    def acc(arr,typ,ct,target=None,bound=False):
        i=len(j['accessors']);a={'bufferView':rawview(arr.tobytes(),target),'componentType':ct,'type':typ,'count':len(arr)}
        if bound:a.update(min=np.atleast_1d(arr.min(0)).tolist(),max=np.atleast_1d(arr.max(0)).tolist())
        j['accessors'].append(a);return i
    # Pack the prop into an extra row of the existing atlas; preserve character texel density.
    vi=j['images'][0]['bufferView'];v=j['bufferViews'][vi]
    original=Image.open(io.BytesIO(b[v['byteOffset']:v['byteOffset']+v['byteLength']])).convert('RGB')
    tex=ball_texture();w,h=original.size;atlas=Image.new('RGB',(w,h+272));atlas.paste(original)
    edge=np.pad(np.asarray(tex),((6,6),(6,6),(0,0)),mode='edge');atlas.paste(Image.fromarray(edge),(0,h+2))
    packed=io.BytesIO();atlas.save(packed,format='PNG');j['images'][0]['bufferView']=rawview(packed.getvalue())
    visited=set()
    for m in j['meshes']:
        for prim in m['primitives']:
            ui=prim['attributes']['TEXCOORD_0']
            if ui in visited:continue
            visited.add(ui);a=j['accessors'][ui];v=j['bufferViews'][a['bufferView']]
            uv=read_accessor(j,b,ui);uv[:,1]*=h/atlas.height
            offset=v['byteOffset']+a.get('byteOffset',0);b[offset:offset+uv.nbytes]=uv.astype('<f4').tobytes()
    base=pose(0,radius)[2];n,uv,faces=ball_mesh();position=base+n*radius
    uv[:,0]=(6.5+uv[:,0]*511)/w;uv[:,1]=(h+8.5+uv[:,1]*255)/atlas.height
    propjoint=len(skin['joints']);node_id=len(nodes);nodes.append({'name':'Basketball','translation':base.tolist()})
    nodes[skin['skeleton']].setdefault('children',[]).append(node_id);skin['joints'].append(node_id)
    ib=read_accessor(j,b,skin['inverseBindMatrices']);prop_ib=np.eye(4,dtype='<f4');prop_ib[:3,3]=-base
    skin['inverseBindMatrices']=acc(np.vstack([ib,prop_ib.T.ravel()]).astype('<f4'),'MAT4',5126)
    joints=np.zeros((len(n),4),'u1');joints[:,0]=propjoint
    weights=np.zeros((len(n),4),'<f4');weights[:,0]=1
    attributes={'POSITION':acc(position.astype('<f4'),'VEC3',5126,34962,True),'NORMAL':acc(n.astype('<f4'),'VEC3',5126,34962),'TEXCOORD_0':acc(uv.astype('<f4'),'VEC2',5126,34962),'JOINTS_0':acc(joints,'VEC4',5121,34962),'WEIGHTS_0':acc(weights,'VEC4',5126,34962)}
    mi=len(j['materials']);j['materials'].append({'name':'Basketball leather','pbrMetallicRoughness':{'baseColorTexture':{'index':0},'metallicFactor':0,'roughnessFactor':.9},'doubleSided':True})
    mesh=len(j['meshes']);j['meshes'].append({'name':'Orange basketball','primitives':[{'attributes':attributes,'indices':acc(faces.astype('<u2').ravel(),'SCALAR',5123,34963),'material':mi}]})
    j['scenes'][0]['nodes'].append(len(nodes));nodes.append({'name':'Basketball mesh','mesh':mesh,'skin':0})
    time=np.unique(np.r_[np.linspace(0,4,121),RELEASE,IMPACT,CATCH]).astype('<f4');timeacc=acc(time,'SCALAR',5126,bound=True)
    nj={nodes[i]['name']:i for i in skin['joints']};tracks={}
    for t in time:
        q,tr,held,chest=pose(float(t),radius);tr['Basketball']=ball(float(t),held,radius,start,end)
        spin=360*smooth(RELEASE,CATCH,t)
        q['Basketball']=chest*rot(x=18+spin,y=-22,z=17)
        for name,value in q.items():tracks.setdefault((name,'rotation'),[]).append(value.as_quat())
        for name,value in tr.items():tracks.setdefault((name,'translation'),[]).append(value)
    sam=[];ch=[]
    for (name,path),data in tracks.items():
        a=np.asarray(data,dtype='<f4')
        if path=='rotation':
            for i in range(1,len(a)):
                if a[i]@a[i-1]<0:a[i]*=-1
        s=len(sam);sam.append({'input':timeacc,'output':acc(a,'VEC4' if path=='rotation' else 'VEC3',5126),'interpolation':'LINEAR'})
        ch.append({'sampler':s,'target':{'node':nj[name],'path':path}})
    j['animations']=[{'name':'Basketball Shoulder Dance','samplers':sam,'channels':ch}]
    # Remove unreachable accessors and buffer views belonging to the old clips or replaced image.
    keepacc=set([skin['inverseBindMatrices']]); keepacc.update(s['input'] for s in sam);keepacc.update(s['output'] for s in sam)
    for m in j['meshes']:
        for pr in m['primitives']:keepacc.add(pr['indices']);keepacc.update(pr['attributes'].values())
    alist=sorted(keepacc);am={old:new for new,old in enumerate(alist)}
    skin['inverseBindMatrices']=am[skin['inverseBindMatrices']]
    for s in sam:s['input']=am[s['input']];s['output']=am[s['output']]
    for m in j['meshes']:
        for pr in m['primitives']:
            pr['indices']=am[pr['indices']];pr['attributes']={k:am[i] for k,i in pr['attributes'].items()}
    j['accessors']=[j['accessors'][i] for i in alist]
    vlist=sorted({a['bufferView'] for a in j['accessors']}|{im['bufferView'] for im in j['images']})
    vm={old:new for new,old in enumerate(vlist)};newbuf=bytearray();newviews=[]
    for i in vlist:
        v=j['bufferViews'][i].copy();start=v.get('byteOffset',0);newbuf.extend(b'\0'*(-len(newbuf)%4));v['byteOffset']=len(newbuf)
        newbuf.extend(b[start:start+v['byteLength']]);newviews.append(v)
    j['bufferViews']=newviews
    for a in j['accessors']:a['bufferView']=vm[a['bufferView']]
    for im in j['images']:im['bufferView']=vm[im['bufferView']]
    newbuf.extend(b'\0'*(-len(newbuf)%4));j['buffers']=[{'byteLength':len(newbuf)}]
    meta=json.dumps(j,ensure_ascii=False,separators=(',',':')).encode();meta+=b' '*(-len(meta)%4)
    destination.write_bytes(struct.pack('<4sII',b'glTF',2,28+len(meta)+len(newbuf))+struct.pack('<I4s',len(meta),b'JSON')+meta+struct.pack('<I4s',len(newbuf),b'BIN\0')+newbuf)
if __name__=='__main__':write()
