"""Read the deliverable GLB and render its embedded animation tracks."""
from rig_animation import OUTPUT,ROOT,load_glb,read_accessor
from render import render
from pathlib import Path
from PIL import Image
from numba import njit
import numpy as np
from scipy.spatial.transform import Rotation
import io,sys,subprocess
@njit(cache=True)
def deform(p,n,ids,w,m):
    points=np.zeros_like(p);normals=np.zeros_like(n)
    for i in range(len(p)):
        for k in range(4):
            weight=w[i,k]
            if weight==0:continue
            mat=m[ids[i,k]]
            for j in range(3):
                points[i,j]+=weight*(mat[j,3]+mat[j,0]*p[i,0]+mat[j,1]*p[i,1]+mat[j,2]*p[i,2])
                normals[i,j]+=weight*(mat[j,0]*n[i,0]+mat[j,1]*n[i,1]+mat[j,2]*n[i,2])
        length=np.sqrt(normals[i,0]**2+normals[i,1]**2+normals[i,2]**2)
        if length>0:
            for j in range(3):normals[i,j]/=length
    return points,normals

class Player:
    def __init__(self,path):
        j,b=load_glb(path)
        def ac(i):return read_accessor(j,b,i)
        def image_from_view(v):return Image.open(io.BytesIO(b[v['byteOffset']:v['byteOffset']+v['byteLength']]))
        self.atlas=np.asarray(image_from_view(j['bufferViews'][j['images'][0]['bufferView']]).convert('RGB'))
        self.joints=j['skins'][0]['joints'];self.names=[j['nodes'][i]['name'] for i in self.joints]
        self.inv=ac(j['skins'][0]['inverseBindMatrices']).reshape(-1,4,4).transpose(0,2,1)
        self.parents=np.full(len(self.joints),-1)
        self.tr=np.asarray([j['nodes'][i]['translation'] for i in self.joints])
        for n,i in enumerate(self.joints):
            for child in j['nodes'][i].get('children',[]):self.parents[self.joints.index(child)]=n
        assert np.all(self.parents<np.arange(len(self.joints)))
        arrays={k:[] for k in ['POSITION','NORMAL','TEXCOORD_0','JOINTS_0','WEIGHTS_0']};fs=[];ids=[];offset=0;self.groups={}
        self.morphs={};self.morph_normals={}
        for node in j['nodes']:
            if 'mesh' not in node:continue
            assert node['skin']==0
            prim=j['meshes'][node['mesh']]['primitives'][0];a=prim['attributes']
            if 'targets' in prim:
                self.morphs[node['name']]=np.array([ac(target['POSITION']) for target in prim['targets']])
                if all('NORMAL' in target for target in prim['targets']):
                    self.morph_normals[node['name']]=np.array([ac(target['NORMAL']) for target in prim['targets']])
            for k in arrays:arrays[k].append(ac(a[k]))
            p=arrays['POSITION'][-1];f=ac(prim['indices']).reshape(-1,3).astype('i')
            assert f.min()>=0 and f.max()<len(p)
            fs.append(f+offset);ids.extend([node['mesh']]*len(f));self.groups[node['name']]=slice(offset,offset+len(p));offset+=len(p)
        self.p,self.n,self.uv,self.indices,self.weights=[np.vstack(arrays[k]) for k in arrays]
        assert self.indices.max()<len(self.joints) and np.allclose(self.weights.sum(1),1,atol=1e-6)
        self.f=np.vstack(fs);self.ids=np.array(ids,dtype='i')
        self.anim={};self.morph_anim={}
        for anim in j['animations']:
            tracks={};morph={}
            for chan in anim['channels']:
                sam=anim['samplers'][chan['sampler']];path=chan['target']['path'];assert path in ('rotation','translation','scale','weights') and sam['interpolation']=='LINEAR'
                t=ac(sam['input']).ravel();q=ac(sam['output']);assert np.all(np.diff(t)>0)
                if path=='weights':
                    mesh_name=j['nodes'][chan['target']['node']]['name']
                    morph[mesh_name]=t,q.reshape(len(t),len(self.morphs[mesh_name]))
                    continue
                if path=='rotation':assert np.allclose(np.linalg.norm(q,axis=1),1,atol=1e-5)
                tracks[self.joints.index(chan['target']['node']),path]=(t,q)
            self.anim[anim['name']]=tracks
            self.morph_anim[anim['name']]=morph
    def morph_weights(self,name,t):
        result={}
        for mesh,(ts,values) in self.morph_anim[name].items():
            i=int(np.clip(np.searchsorted(ts,t,side='right')-1,0,len(ts)-2))
            f=float(np.clip((t-ts[i])/(ts[i+1]-ts[i]),0,1))
            result[mesh]=values[i]*(1-f)+values[i+1]*f
        return result
    def matrices(self,name,t):
        worlds=np.repeat(np.eye(4)[None],len(self.joints),axis=0)
        worlds[:,:3,3]=self.tr
        scales=np.ones((len(self.joints),3))
        for (i,path),(ts,qs) in self.anim[name].items():
            h=np.clip(np.searchsorted(ts,t,side='right')-1,0,len(ts)-2)
            u=np.clip((t-ts[h])/(ts[h+1]-ts[h]),0,1)
            if path=='translation':
                worlds[i,:3,3]=qs[h]*(1-u)+qs[h+1]*u
                continue
            if path=='scale':
                scales[i]=qs[h]*(1-u)+qs[h+1]*u
                continue
            # glTF LINEAR rotation samplers follow the shortest spherical arc.
            q0,q1=qs[h],qs[h+1];cos=float(q0@q1)
            if cos<0:q1=-q1;cos=-cos
            if cos>.9995:q=q0*(1-u)+q1*u
            else:
                theta=np.arccos(np.clip(cos,-1,1));q=(np.sin((1-u)*theta)*q0+np.sin(u*theta)*q1)/np.sin(theta)
            worlds[i,:3,:3]=Rotation.from_quat(q).as_matrix()
        worlds[:,:3,:3]*=scales[:,None,:]
        for i,p in enumerate(self.parents):
            if p>=0:worlds[i]=worlds[p]@worlds[i]
        return worlds,(worlds@self.inv).astype('f')
    def frame(self,name,t):
        _,matrices=self.matrices(name,t)
        base=self.p.copy() if self.morphs else self.p
        normal=self.n.copy() if self.morph_normals else self.n
        for mesh,w in self.morph_weights(name,t).items():
            base[self.groups[mesh]]+=np.einsum('i,ijk->jk',w,self.morphs[mesh])
            if mesh in self.morph_normals:
                normal[self.groups[mesh]]+=np.einsum('i,ijk->jk',w,self.morph_normals[mesh])
        ps,ns=deform(base,normal,self.indices,self.weights,matrices)
        return ps,ns,self.uv,self.f,self.ids,self.atlas

def main():
    model=Player(OUTPUT)
    static=model.frame('Idle',0)[0];assert np.max(abs(static-model.p))<1e-5
    assert np.max(abs(model.frame('Wave',4.)[0]-model.p))<1e-5
    wave=model.frame('Wave',1.4)[0];hand=model.groups['Mitten palm -1'];foot=model.groups['Mary Jane -1']
    shift=(wave[hand]-static[hand]).mean(0)
    assert shift[1]>.30 and shift[2]>.16 and np.max(abs(wave[foot]-static[foot]))<1e-5
    print('Rest verified. Raised hand delta:',shift,flush=True)
    if '--check' in sys.argv:
        angles=[(-12,.0),(-12,.58),(-12,1.4),(-12,1.75),(-12,2.15),(35,1.75)]
        sheet=Image.new('RGB',(6*270,340))
        for i,(a,t) in enumerate(angles):
            im=render(model.frame('Wave',t),a,1,(400,500));sheet.paste(im.resize((270,338),Image.Resampling.LANCZOS),(270*i,0))
        sheet.save(ROOT/'制作'/'挥手动作检查.jpg');return
    width,height,fps=656,736,24
    cmd=['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{height}','-r',str(fps),'-i','pipe:0','-an','-c:v','libx264','-threads','3','-crf','17','-preset','fast','-pix_fmt','yuv420p','-movflags','+faststart',str(OUTPUT.parent/'待机与挥手.mp4')]
    proc=subprocess.Popen(cmd,stdin=subprocess.PIPE)
    try:
        for clip,num in [('Idle',84),('Wave',96)]:
            for i in range(num):
                im=render(model.frame(clip,i/fps),-11,1,(width,height))
                proc.stdin.write(np.asarray(im).tobytes())
    finally:proc.stdin.close()
    assert proc.wait()==0
if __name__=='__main__':main()
