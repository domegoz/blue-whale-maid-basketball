"""Render the exported take on a pastel studio court with its original audio."""
from pathlib import Path
from fractions import Fraction
import subprocess
import sys
import tempfile

import av
import numpy as np
from PIL import Image, ImageDraw

from retarget import CLIP, GLB, HERE, OUT, RADIUS
from preview_rig import Player
import moderngl as gl

WIDTH, HEIGHT, FPS, COUNT = 1920,1080,60,2958
BEGIN = 0.
MP4 = OUT / '蓝鲸女仆_自我介绍与篮球舞_1080p60.mp4'
SOURCE = HERE/'自我介绍与篮球舞_音轨.flac'

MESH_VERTEX = """
#version 330
uniform mat4 bones[40];
uniform vec4 expressions[32];
uniform sampler2D shapeNormals;
uniform float aspect;
uniform vec2 framing;
in vec3 pos;
in vec3 normal;
in vec2 texcoord;
in vec4 joints;
in vec4 weights;
in vec3 offsetA;
in vec3 offsetB;
in vec3 offsetC;
in vec3 offsetD;
in vec3 offsetE;
in vec3 offsetF;
in vec3 offsetG;
in vec3 offsetH;
in float shapeGroup;
out vec3 n;
out vec2 uv;
out float cameraZ;
const float tilt=.105;
void main(){
    mat4 m = bones[int(joints.x)]*weights.x+bones[int(joints.y)]*weights.y+
             bones[int(joints.z)]*weights.z+bones[int(joints.w)]*weights.w;
    int slot=int(shapeGroup)*2;
    vec4 a=expressions[slot],b=expressions[slot+1];
    vec3 sculpt=pos+a.x*offsetA+a.y*offsetB+a.z*offsetC+a.w*offsetD
                   +b.x*offsetE+b.y*offsetF+b.z*offsetG+b.w*offsetH;
    vec3 world=(m*vec4(sculpt,1)).xyz;
    vec3 p=world-vec3(0,framing.y,.10);
    float y=p.y*cos(tilt)-p.z*sin(tilt);
    cameraZ=p.y*sin(tilt)+p.z*cos(tilt);
    gl_Position=vec4(2.*p.x/(framing.x*aspect),2.*y/framing.x,-cameraZ/12.,1);
    vec3 localNormal=normal;
    if(slot!=0){
        int index=gl_VertexID*8;
        for(int i=0;i<8;i++){
            float weight=expressions[slot+i/4][i%4];
            localNormal+=weight*texelFetch(shapeNormals,ivec2((index+i)%1024,(index+i)/1024),0).xyz;
        }
    }
    n=mat3(m)*localNormal;
    uv=texcoord;
}
"""
MESH_FRAGMENT = """
#version 330
uniform sampler2D atlas;
in vec2 uv;
in vec3 n;
in float cameraZ;
layout(location=0) out vec4 colour;
layout(location=1) out float depth;
void main(){
    vec3 N=normalize(n);
    vec3 light=normalize(vec3(-.38,.67,.78));
    float shade=.888+.14*max(dot(N,light),0.)+.025*max(N.y,0.);
    vec3 rgb=texture(atlas,uv).rgb*shade;
    float rim=pow(1.-abs(clamp(N.z,-1.,1.)),3.)*.032;
    colour=vec4(rgb+rim*vec3(.64,.72,1.),1);
    depth=cameraZ+6.;
}
"""
QUAD = """
#version 330
out vec2 uv;
void main(){
    vec2 p=vec2((gl_VertexID<<1)&2,gl_VertexID&2);
    uv=p;
    gl_Position=vec4(p*2.-1.,0,1);
}
"""
COMPOSE = """
#version 330
in vec2 uv;
out vec4 result;
uniform sampler2D surface;
uniform sampler2D depthMap;
uniform float aspect;
uniform vec2 framing;
uniform vec2 pixel;
uniform vec3 footA,footB,ball,hip;
uniform float showBall;
const float tilt=.105;
float hash(vec2 p){return fract(sin(dot(p,vec2(127.1,311.7)))*43758.5453);}
float line(float d,float w){return 1.-smoothstep(w,w+fwidth(d)*1.25,abs(d));}
float contact(vec2 p,vec3 loc,vec2 spread){
    vec2 d=(p-loc.xz)/spread;
    return exp(-dot(d,d));
}
vec3 stage(){
    vec2 proj=(uv-.5)*vec2(framing.x*aspect,framing.x);
    float z=.10+(-framing.y*cos(tilt)-proj.y)/sin(tilt);
    vec2 floorPos=vec2(proj.x,z);
    vec3 wall=mix(vec3(.848,.849,.929),vec3(.955,.948,.977),clamp(uv.y+.10,0.,1.));
    wall+=.028*exp(-dot((uv-vec2(.44,.63))*1.6,(uv-vec2(.44,.63))*1.6));
    float stripe=uv.x+.66*uv.y;
    float blue=smoothstep(.480,.482,stripe)*(1.-smoothstep(.533,.535,stripe));
    float mint=smoothstep(.415,.417,stripe)*(1.-smoothstep(.48,.482,stripe));
    wall=mix(wall,vec3(.57,.57,.85),blue*.69);
    wall=mix(wall,vec3(.58,.83,.86),mint*.77);
    // Light scattered from the floor onto the base of the cyclorama.
    wall-=.028*exp(-14.*abs(z+5.45));
    vec3 floorCol=mix(vec3(.833,.750,.689),vec3(.947,.866,.780),clamp((z+5.45)/8.,0.,1.));
    float row=floor((z+5.45)/.43);
    float offset=fract(row*.618)*2.63;
    float tile=floor((floorPos.x+offset)/2.63);
    float grain=.0025*sin(floorPos.x*26.+4.*sin(row*10.+z*5.))+.0015*sin(z*215.+floorPos.x*2.);
    floorCol+=vec3(1.,.84,.69)*(.014*(hash(vec2(row,tile))-.5)+grain);
    float joinZ=min(fract((z+5.45)/.43),1.-fract((z+5.45)/.43))*.43;
    float joinX=min(fract((floorPos.x+offset)/2.63),1.-fract((floorPos.x+offset)/2.63))*2.63;
    floorCol-=.024*(max(line(joinZ,.0025),line(joinX,.002)));
    float arc=length((floorPos-vec2(-2.46,-.51)))-2.40;
    float marking=max(line(arc,.013),line(z-1.68,.011));
    floorCol=mix(floorCol,vec3(.99,.96,.89),marking*.60);
    float shade=.13*contact(floorPos,hip+vec3(.11,0,-.20),vec2(.49,.59));
    shade+=.15*contact(floorPos,footA,vec2(.121,.18))*exp(-6.*max(footA.y-.09,0.));
    shade+=.15*contact(floorPos,footB,vec2(.121,.18))*exp(-6.*max(footB.y-.09,0.));
    shade+=showBall*.18*contact(floorPos,ball,vec2(.15,.21)+max(ball.y-.217,0.)*.12)*exp(-2.9*max(ball.y-.217,0.));
    floorCol=mix(floorCol,vec3(.34,.30,.39),min(.55,shade));
    return mix(wall,floorCol,smoothstep(-5.5,-5.40,z));
}
vec4 toon(vec2 p){
    vec4 s=texture(surface,p);
    float z=texture(depthMap,p).r;
    float stepZ=min(min(texture(depthMap,p+vec2(pixel.x,0)).r,texture(depthMap,p-vec2(pixel.x,0)).r),
                    min(texture(depthMap,p+vec2(0,pixel.y)).r,texture(depthMap,p-vec2(0,pixel.y)).r));
    float ink=smoothstep(.036,.18,z-stepZ)*s.a;
    return vec4(mix(s.rgb,vec3(.128,.114,.206),ink*.30),s.a);
}
void main(){
    vec4 s=(toon(uv+pixel*vec2(-.34,-.34))+toon(uv+pixel*vec2(.34,-.34))+
            toon(uv+pixel*vec2(-.34,.34))+toon(uv+pixel*vec2(.34,.34)))*.25;
    vec3 rgb=stage()*(1.-s.a)+s.rgb;
    result=vec4(rgb,1);
}
"""


class Court:
    def __init__(self,w,h):
        self.player=Player(GLB)
        try:
            ctx=gl.create_standalone_context()
        except Exception:
            if not sys.platform.startswith('linux'):
                raise
            ctx=gl.create_standalone_context(backend='egl')
        self.ctx=ctx
        self.w,self.h=w,h
        self.mesh=mesh=ctx.program(vertex_shader=MESH_VERTEX,fragment_shader=MESH_FRAGMENT)
        self.finish=finish=ctx.program(vertex_shader=QUAD,fragment_shader=COMPOSE)
        self.quad=ctx.vertex_array(finish,[])
        self.mesh['aspect']=self.finish['aspect']=w/h
        self.finish['pixel']=(1/w,1/h)
        self.mesh['atlas']=0
        self.mesh['shapeNormals']=3
        self.finish['surface']=1
        self.finish['depthMap']=2
        m=self.player
        rows=(len(m.p)*8+1023)//1024
        normal_table=np.zeros((rows*1024,3),'f2')
        normal_slots=normal_table[:len(m.p)*8].reshape(-1,8,3)
        for name,targets in m.morph_normals.items():
            normal_slots[m.groups[name],:len(targets)]=targets.transpose(1,0,2)
        self.normal_table=ctx.texture((1024,rows),3,normal_table.tobytes(),dtype='f2',alignment=1)
        self.normal_table.filter=(gl.NEAREST,gl.NEAREST)
        offsets=np.zeros((len(m.p),8,3),'f4')
        slots=np.zeros((len(m.p),1),'f4')
        self.shape_indices={name:i for i,name in enumerate(m.morphs,1)}
        assert len(self.shape_indices)<16
        for name,index in self.shape_indices.items():
            shapes=m.morphs[name]
            offsets[m.groups[name],:len(shapes)]=shapes.transpose(1,0,2)
            slots[m.groups[name]]=index
        arrays=[m.p,m.n,m.uv,m.indices,m.weights,offsets.reshape(-1,24),slots]
        vertices=np.ascontiguousarray(np.hstack(arrays),dtype='f4')
        layout='3f 3f 2f 4f 4f '+'3f '*8+'1f'
        self.obj=ctx.vertex_array(mesh,[(ctx.buffer(vertices),layout,'pos','normal','texcoord','joints','weights',*['offset'+s for s in 'ABCDEFGH'],'shapeGroup')],
                                  ctx.buffer(np.ascontiguousarray(m.f,'u4')),index_element_size=4)
        self.atlas=ctx.texture((m.atlas.shape[1],m.atlas.shape[0]),3,m.atlas.tobytes(),alignment=1)
        self.atlas.build_mipmaps()
        self.atlas.filter=(gl.LINEAR_MIPMAP_LINEAR,gl.LINEAR)
        self.atlas.repeat_x=self.atlas.repeat_y=False
        self.atlas.anisotropy=4
        s=(round(w*1.25),round(h*1.25))
        self.colour=ctx.texture(s,4)
        self.depth_map=ctx.texture(s,1,dtype='f4')
        self.depth_map.filter=(gl.NEAREST,gl.NEAREST)
        self.colour.repeat_x=self.colour.repeat_y=False
        self.depth_map.repeat_x=self.depth_map.repeat_y=False
        self.scene=ctx.framebuffer([self.colour,self.depth_map],ctx.depth_renderbuffer(s))
        self.screen=ctx.simple_framebuffer((w,h),components=3)

    def frame(self,time):
        c=self.ctx
        u=np.clip((time-7.50)/2.65,0,1)
        closeup=1-u*u*(3-2*u)
        # Follow the bow with a small downward tilt, then ease to the full-court shot.
        v=np.clip((time-.30)/.80,0,1)
        bow=1-v*v*(3-2*v)
        framing=(2.86-1.05*closeup,1.23+.34*closeup-.24*bow)
        self.mesh['framing']=self.finish['framing']=framing
        world,skin=self.player.matrices(CLIP,time)
        mats=np.tile(np.eye(4,dtype='f4'),(40,1,1))
        mats[:len(skin)]=skin.transpose(0,2,1)
        self.mesh['bones'].write(mats.tobytes())
        coeff=np.zeros((16,8),'f4')
        for name,w in self.player.morph_weights(CLIP,time).items():
            coeff[self.shape_indices[name],:len(w)]=w
        self.mesh['expressions'].write(coeff.tobytes())
        self.scene.use()
        self.scene.clear(0,0,0,0)
        c.enable(gl.DEPTH_TEST)
        self.atlas.use(0)
        self.normal_table.use(3)
        self.obj.render()
        c.disable(gl.DEPTH_TEST)
        for uni,bone in [('footA','Foot.L'),('footB','Foot.R'),('hip','Hips'),('ball','Basketball')]:
            m=world[self.player.names.index(bone)]
            self.finish[uni].value=tuple(m[:3,3])
        self.finish['showBall']=float(np.linalg.norm(m[:3,0])>.01)
        self.screen.use()
        self.colour.use(1)
        self.depth_map.use(2)
        self.quad.render(gl.TRIANGLES,vertices=3)
        return self.screen.read(components=3,alignment=1)


def review():
    c=Court(600,338)
    times=[9.0,10.75,11.05,13.77,14.13,15.40,17.78,20.1,21.83,23.33,25.23,27.10,
           28.90,30.67,32.12,33.87,35.07,38.10,40.95,42.12,43.84,45.10,47.22,48.92]
    panel=Image.new('RGB',(4*320,6*200),'#25242d')
    draw=ImageDraw.Draw(panel)
    for i,t in enumerate(times):
        image=Image.frombytes('RGB',(600,338),c.frame(t-BEGIN)).transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        x,y=i%4*320,i//4*200
        panel.paste(image.resize((320,180),Image.Resampling.LANCZOS),(x,y))
        draw.text((x+5,y+182),f'{t:05.2f}',fill='white')
    panel.save(HERE/'成片动作检查.jpg',quality=95)


def original_audio(stream):
    rate=48000
    length=round(COUNT/FPS*rate)
    first=round(BEGIN*rate)
    samples=np.zeros((2,length),'f4')
    covered=np.zeros(length,bool)
    with av.open(str(SOURCE)) as source:
        res=av.AudioResampler(format='fltp',layout='stereo',rate=rate)
        for frame in source.decode(audio=0):
            for f in res.resample(frame):
                ix=round(f.time*rate)-first
                lo,hi=max(0,ix),min(length,ix+f.samples)
                if hi>lo:
                    samples[:,lo:hi]=f.to_ndarray()[:,lo-ix:hi-ix]
                    covered[lo:hi]=True
            if frame.time>BEGIN+COUNT/FPS+.05: break
    if np.count_nonzero(~covered)>.025*rate:
        raise ValueError('Unexpected gap in source audio')
    for seconds,fade_in in [(.045,True),(.14,False)]:
        count=round(rate*seconds)
        ramp=np.linspace(0,1,count,dtype='f4')
        if fade_in: samples[:,:count]*=ramp
        else: samples[:,-count:]*=ramp[::-1]
    packets=[]
    for ix in range(0,length,1024):
        f=av.AudioFrame.from_ndarray(np.ascontiguousarray(samples[:,ix:ix+1024]),format='fltp',layout='stereo')
        f.sample_rate=rate
        f.pts,f.time_base=ix,Fraction(1,rate)
        packets.extend(stream.encode(f))
    packets.extend(stream.encode(None))
    return packets


def render():
    c=Court(WIDTH,HEIGHT)
    temp=MP4.with_suffix('.rendering.mp4')
    cmd=['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24',
         '-s',f'{WIDTH}x{HEIGHT}','-r',str(FPS),'-i','pipe:0','-map','0:v','-an','-vf','vflip',
         '-c:v','libx264','-threads','6','-preset','veryfast','-crf','17','-pix_fmt','yuv420p',
         '-f','mp4','-movflags','frag_keyframe+empty_moov+default_base_moof','pipe:1']
    with tempfile.TemporaryFile(dir=HERE) as muxed:
        proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=muxed)
        try:
            for i in range(COUNT):
                proc.stdin.write(c.frame(i/FPS))
                if i and i%300==0: print('Rendered',i,'/',COUNT,flush=True)
        finally:
            proc.stdin.close()
            code=proc.wait()
        if code: raise RuntimeError(f'ffmpeg exited with {code}')
        muxed.seek(0)
        with av.open(muxed) as src, av.open(str(temp),'w',options={'movflags':'+faststart'}) as dst:
            video=src.streams.video[0]
            target=dst.add_stream_from_template(video)
            audio=dst.add_stream('aac',rate=48000)
            audio.layout='stereo'
            audio.bit_rate=192000
            dst.start_encoding()
            packets=original_audio(audio)
            for packet in src.demux(video):
                if packet.dts is not None:
                    packet.pts-=video.start_time or 0
                    packet.dts-=video.start_time or 0
                    packet.stream=target
                    packets.append(packet)
            for packet in sorted(packets,key=lambda p:p.dts*p.time_base):
                dst.mux(packet)
    temp.replace(MP4)
    print(MP4,flush=True)


if __name__=='__main__':
    if '--review' in sys.argv: review()
    else: render()
