"""Add a reusable glTF armature, painted weights and two looping actions."""
from pathlib import Path
import json
import struct
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'蓝鲸女仆.glb'
OUTPUT=ROOT/'骨骼动画'/'蓝鲸女仆_骨骼动画.glb'

def load_glb(path):
    raw=path.read_bytes()
    magic,version,total,jsize,jtype=struct.unpack_from('<4sIIII',raw)
    assert magic==b'glTF' and version==2 and total==len(raw) and jtype==0x4e4f534a
    j=json.loads(raw[20:20+jsize]);length,kind=struct.unpack_from('<II',raw,20+jsize)
    assert kind==0x004e4942 and length==len(raw)-28-jsize
    return j,bytearray(raw[28+jsize:])

def read_accessor(doc,buf,i):
    a=doc['accessors'][i];v=doc['bufferViews'][a['bufferView']]
    dt={5121:'u1',5123:'<u2',5125:'<u4',5126:'<f4'}[a['componentType']]
    dims={'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4,'MAT4':16}[a['type']]
    data=np.frombuffer(buf,dt,count=a['count']*dims,offset=v.get('byteOffset',0)+a.get('byteOffset',0)).reshape(-1,dims).copy()
    if a.get('normalized'):data=data.astype('f')/np.iinfo(dt).max
    return data

def smooth(a,b,x):
    t=np.clip((x-a)/(b-a),0,1)
    return t*t*(3-2*t)

class Armature:
    def __init__(self):
        self.names=[];self.parent=[];self.anchors=[];self.index={}
    def bone(self,name,parent,x,y,z):
        self.index[name]=len(self.names);self.names.append(name)
        self.parent.append(-1 if parent is None else self.index[parent]);self.anchors.append([x,y,z])
    def point(self,name):return np.asarray(self.anchors[self.index[name]])

def skeleton():
    r=Armature();add=r.bone
    add('Root',None,0,0,0);add('Hips','Root',0,.70,.14)
    add('Spine','Hips',0,.754,.133);add('Chest','Spine',0,.858,.124)
    add('Neck','Chest',0,.958,.145);add('Head','Neck',0,1.016,.14)
    for side,k in [('R',-1),('L',1)]:
        add('Clavicle.'+side,'Chest',k*.067,.922,.131)
        add('UpperArm.'+side,'Clavicle.'+side,k*.190,.921,.122)
        add('Forearm.'+side,'UpperArm.'+side,k*.286,.757,.139)
        add('Hand.'+side,'Forearm.'+side,k*.384,.581,.154)
        add('Thigh.'+side,'Hips',k*.123,.678,.122)
        add('Shin.'+side,'Thigh.'+side,k*.137,.311,.133)
        add('Foot.'+side,'Shin.'+side,k*.137,.091,.144)
    for side,x,z in [('R',-.47,-.20),('C',0,-.52),('L',.47,-.20)]:
        add('HairUpper.'+side,'Head',x,1.22,z);add('HairTip.'+side,'HairUpper.'+side,x*.99,.826,z-.045)
    add('Cowlick','Head',-.017,2.006,.166)
    add('Skirt','Hips',0,.713,.15);add('SkirtHem','Skirt',0,.44,.15)
    add('TailBase','Hips',.135,.457,-.095);add('TailMid','TailBase',.365,.399,-.234)
    add('TailFlukes','TailMid',.548,.457,-.105)
    return r

def skin_weights(name,p,r):
    x,y,z=p.T;w=np.zeros((len(p),len(r.names)))
    def assign(b,amt=1.):w[:,r.index[b]]+=amt
    side='L' if x.mean()>0 else 'R'
    if name.startswith(('Crown and','Hair lining','Rounded hem','Cheek curl','Surf curl','Rear cascade')):
        blend=smooth(1.32,1.065,y);tip=smooth(.98,.64,y)
        center=1-smooth(.12,.42,abs(x))
        assign('Head',1-blend)
        for s,portion in [('C',center),('R',(1-center)*(x<0)),('L',(1-center)*(x>=0))]:
            assign('HairUpper.'+s,blend*portion*(1-tip));assign('HairTip.'+s,blend*portion*tip)
    elif name.startswith('Crown cowlick'):
        tip=smooth(2.00,2.115,y);assign('Head',1-tip);assign('Cowlick',tip)
    elif name.startswith(('Face ','Ear ','Bangs','Pleated lace','Lace facing','Lace scallop','Indigo headband','Satin binding','Wing-shaped','Temple ')):
        assign('Head')
    elif name=='Neck':
        h=smooth(.971,1.043,y);assign('Neck',1-h);assign('Head',h)
    elif name.startswith('Puffed sleeve'):
        suffix='.'+side
        sh=r.point('UpperArm'+suffix);el=r.point('Forearm'+suffix)
        axis=(el-sh)/np.linalg.norm(el-sh);along=(p-el)@axis
        fore=smooth(-.071,.070,along);clav=smooth(-.12,-.204,along)*.48
        assign('Forearm'+suffix,fore);assign('UpperArm'+suffix,(1-fore)*(1-clav));assign('Clavicle'+suffix,(1-fore)*clav)
    elif name.startswith('Cuff layer'):assign('Forearm.'+side)
    elif name.startswith(('Mitten','Fingertip','Thumb')):assign('Hand.'+side)
    elif name.startswith('Stocking'):
        foot=smooth(.131,.083,y);assign('Shin.'+side,1-foot);assign('Foot.'+side,foot)
    elif name.startswith(('Mary Jane','Shoe opening','Instep strap','Shoe clasp')):assign('Foot.'+side)
    elif name.startswith(('Midnight','Scalloped ivory','Skirt underside','Whale apron','Skirt ribbon')):
        k=smooth(.733,.619,y);hem=smooth(.493,.282,y)
        assign('Hips',1-k);assign('Skirt',k*(1-hem));assign('SkirtHem',k*hem)
    elif name.startswith(('Fitted bodice','Ruffled shoulder','Tied apron')):
        s=smooth(.722,.818,y);c=smooth(.787,.910,y)
        assign('Hips',1-s);assign('Spine',s*(1-c));assign('Chest',s*c)
    elif name.startswith(('Sailor throat','Silk bow centre','Cameo','Ocean cabochon')):assign('Chest')
    elif name.startswith('Whale tail'):
        # Keep the fluke roots and their connection to the stem on the same gradient.
        mid=smooth(.23,.436,x);fan=smooth(.46,.572,x)
        assign('TailBase',1-mid);assign('TailMid',mid*(1-fan));assign('TailFlukes',mid*fan)
    else:raise ValueError('Unassigned surface: '+name)
    assert np.allclose(w.sum(axis=1),1)
    ids=np.argsort(-w,axis=1,kind='stable')[:,:4].astype('u1')
    values=np.take_along_axis(w,ids,axis=1);assert np.allclose(values.sum(axis=1),1)
    q=np.rint(values*65535).astype(np.int32);q[:,0]+=65535-q.sum(axis=1);ids[q==0]=0
    return ids,q.astype('<u2')

def rot(x=0,y=0,z=0):return Rotation.from_euler('xyz',[x,y,z],degrees=True)
def align(a,b):
    a=a/np.linalg.norm(a);b=b/np.linalg.norm(b);q=np.r_[np.cross(a,b),1+np.dot(a,b)]
    return Rotation.from_quat(q/np.linalg.norm(q))
def short_blend(rotation,amount):return Rotation.from_rotvec(rotation.as_rotvec()*amount)

def pose(r,clip,t,duration):
    phase=t/duration*2*np.pi
    s=np.sin(phase);c=1-np.cos(phase)
    # A neutral first/last frame permits both clean looping and clip transitions.
    amt=float(smooth(.17,.99,t)*(1-smooth(2.86,3.79,t))) if clip=='Wave' else 0.
    wind=np.sin(2*np.pi*(t-1.02)/.72)
    pre=float(smooth(.04,.29,t)*(1-smooth(.29,.58,t))) if clip=='Wave' else 0.
    result={'Spine':rot(x=.9*s+1.35*pre,z=.55*s+.85*amt),
            'Chest':rot(x=-.9*s-2.4*amt,y=1.3*s+6.2*amt,z=-.5*s+.7*amt),
            'Neck':rot(x=.65*s-.8*amt,z=.65*s-1.7*amt),
            'Head':rot(x=.8*s+1.1*amt,y=2*s-6.7*amt,z=1*s-4.1*amt),
            'Cowlick':rot(x=1.2*c,z=2*s+2.6*amt*wind),
            'Skirt':rot(x=.45*s,z=.38*s+.35*amt),
            'SkirtHem':rot(x=.52*s,z=.6*s+.45*amt),
            'TailBase':rot(y=4.6*s,z=1.6*c+1.5*amt),
            'TailMid':rot(x=2.1*s,y=5.0*s+2.8*amt*wind,z=2*c),
            'TailFlukes':rot(x=3.3*s,y=4.5*s,z=2.6*s+1.9*amt*wind)}
    for side,sgn in [('R',-1),('L',1)]:
        suf='.'+side
        result['Clavicle'+suf]=rot(x=-.7*s,y=sgn*.8*s,z=-sgn*(.65*s+(4*amt if side=='R' else 0)))
        if side=='R' and clip=='Wave':
            sh=r.point('UpperArm.R');el=r.point('Forearm.R');wr=r.point('Hand.R')
            # Elbow comes forward before the palm approaches the lower cheek.
            upper=short_blend(align(el-sh,np.array([-.130,-.043,.135])),amt)
            fore=short_blend(align(wr-el,np.array([-.085+.099*wind,.145,.103+.014*wind])),amt)
            result['UpperArm.R']=upper*rot(z=-1.15*s+3.0*pre)
            result['Forearm.R']=upper.inv()*fore
            result['Hand.R']=rot(x=-8*amt,y=9*amt*wind,z=amt*(4+12*wind))
        else:
            result['UpperArm'+suf]=rot(x=-1.4*s-2*amt,y=sgn*s,z=sgn*(1.45*s+2.3*amt))
            result['Forearm'+suf]=rot(x=-1.6*s-.9*amt,z=sgn*.8*s)
            result['Hand'+suf]=rot(z=sgn*.7*s)
    for side,m in [('L',1),('C',0),('R',-1)]:
        result['HairUpper.'+side]=rot(x=-.8*s+.25*c,y=-1.3*s+.7*amt,z=-1.3*s+.40*m*c+1.4*amt)
        result['HairTip.'+side]=rot(x=-1.1*s+.39*c,y=-1.6*s+.5*c,z=-1.6*s+.65*m*c+1.2*amt)
    return result

def make():
    doc,buf=load_glb(SOURCE);r=skeleton()
    def add(data,typ,ct,target=None,normalized=False,bounds=False):
        while len(buf)%4:buf.append(0)
        v={'buffer':0,'byteOffset':len(buf),'byteLength':data.nbytes}
        if target:v['target']=target
        vi=len(doc['bufferViews']);doc['bufferViews'].append(v);buf.extend(data.tobytes())
        a={'bufferView':vi,'componentType':ct,'count':len(data),'type':typ}
        if normalized:a['normalized']=True
        if bounds:a.update(min=[float(data.min())],max=[float(data.max())])
        doc['accessors'].append(a);return len(doc['accessors'])-1
    orig=doc['nodes'];group={c:n['name'] for n in orig for c in n.get('children',[])}
    nodes=[];roots=[]
    for i,n in enumerate(orig):
        if 'mesh' not in n:continue
        assert not any(k in n for k in ('translation','rotation','scale','matrix'))
        n=dict(n,skin=0,extras={'partGroup':group[i]});roots.append(len(nodes));nodes.append(n)
        for prim in doc['meshes'][n['mesh']]['primitives']:
            a=prim['attributes'];p=read_accessor(doc,buf,a['POSITION']);ids,weights=skin_weights(n['name'],p,r)
            a['JOINTS_0']=add(ids,'VEC4',5121,34962)
            a['WEIGHTS_0']=add(weights,'VEC4',5123,34962,True)
    first=len(nodes);joint_ids=list(range(first,first+len(r.names)));abs_pos=np.asarray(r.anchors)
    for i,name in enumerate(r.names):
        parent=r.parent[i];local=abs_pos[i] if parent<0 else abs_pos[i]-abs_pos[parent]
        node={'name':name,'translation':local.tolist()};nodes.append(node)
        if parent>=0:nodes[first+parent].setdefault('children',[]).append(first+i)
    roots.append(first);doc['nodes']=nodes;doc['scenes']=[{'name':'Blue Whale Maid · rig','nodes':roots}];doc['scene']=0
    inverses=np.repeat(np.eye(4,dtype='<f4')[None],len(r.names),axis=0);inverses[:,:3,3]=-abs_pos
    doc['skins']=[{'name':'Blue Whale Maid · skeleton','skeleton':first,'joints':joint_ids,'inverseBindMatrices':add(inverses.transpose(0,2,1).copy(),'MAT4',5126)}]
    doc['animations']=[]
    for clip,duration in [('Idle',3.5),('Wave',4.)]:
        ts=np.linspace(0,duration,round(duration*24)+1).astype('<f4');time_id=add(ts,'SCALAR',5126,bounds=True)
        frames=[pose(r,clip,float(t),duration) for t in ts];samplers=[];channels=[]
        for name in frames[0]:
            q=np.array([f[name].as_quat() for f in frames],dtype='<f4')
            for i in range(1,len(q)):
                if q[i-1]@q[i]<0:q[i]*=-1
            if np.max(abs(q[:,:3]))<.00001:continue
            channels.append({'sampler':len(samplers),'target':{'node':first+r.index[name],'path':'rotation'}})
            samplers.append({'input':time_id,'output':add(q,'VEC4',5126),'interpolation':'LINEAR'})
        doc['animations'].append({'name':clip,'samplers':samplers,'channels':channels})
    while len(buf)%4:buf.append(0)
    doc['buffers']=[{'byteLength':len(buf)}]
    meta=json.dumps(doc,ensure_ascii=False,separators=(',',':')).encode();meta+=b' '*(-len(meta)%4)
    b=struct.pack('<4sII',b'glTF',2,28+len(meta)+len(buf))+struct.pack('<I4s',len(meta),b'JSON')+meta+struct.pack('<I4s',len(buf),b'BIN\0')+buf
    tmp=OUTPUT.with_suffix('.part');tmp.write_bytes(b);tmp.replace(OUTPUT)
    print(len(r.names),'bones /',first,'meshes /',len(b),'bytes')
if __name__=='__main__':make()

