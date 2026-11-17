from pathlib import Path
import base64
from PIL import Image
import io
R=Path(__file__).resolve().parents[1]
html="""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>蓝鲸女仆 · 3D 模型</title>
<style>[hidden]{display:none!important}*{box-sizing:border-box}body{margin:0;background:#f7f5f2;color:#32334a;font-family:"Microsoft YaHei UI",system-ui,sans-serif}main{height:100dvh;display:flex;flex-direction:column;align-items:center}header{width:min(95%,1040px);padding:18px 6px 10px;display:flex;align-items:baseline;justify-content:space-between;gap:12px}h1{font-size:20px;letter-spacing:.14em;font-weight:600;margin:0}#status{font-size:12px;color:#676779}.stage{position:relative;width:100%;flex:1;min-height:0}.stage>*{position:absolute;width:100%;height:100%;object-fit:contain;inset:0}canvas{touch-action:none;cursor:grab}canvas:active{cursor:grabbing}footer{padding:10px 12px 24px;display:flex;gap:6px;flex-wrap:wrap;justify-content:center}button{border:1px solid #d8d7df;color:#47495e;background:#fff9;border-radius:18px;min-width:48px;padding:7px 12px;font:inherit;font-size:13px;cursor:pointer}button:hover,button:focus-visible{border-color:#767caf;color:#393a71}button[aria-pressed=true]{background:#515887;color:#fff;border-color:#515887}@media(max-width:550px){header{padding:12px 0 4px}h1{font-size:17px}footer{padding-bottom:12px}#status{max-width:58%;text-align:right}}</style>
<main><header><h1>蓝鲸女仆</h1><span id="status" role="status">正在载入模型…</span></header><div class="stage"><img id="fallback" src="@@POSTER@@" alt="角色模型预览"><canvas aria-label="可拖动旋转的三维模型" tabindex="0"></canvas></div><footer aria-label="视角控制"><button data-a="0">正面</button><button data-a="90">侧面</button><button data-a="180">背面</button><button id="spin" aria-pressed="false">自动旋转</button><button id="reset">复位</button></footer></main>
<script type="application/octet-stream" id="asset">@@MODEL@@</script>
<script>
(async()=>{
const canvas=document.querySelector('canvas'),status=document.querySelector('#status'),fallback=document.querySelector('#fallback');
try{
 const gl=canvas.getContext('webgl2',{alpha:false,antialias:true,powerPreference:'high-performance'});if(!gl)throw Error('此浏览器未启用 WebGL 2，当前显示静态预览');
 const vert=`#version 300 es
 layout(location=0) in vec3 p;layout(location=1) in vec3 n;layout(location=2) in vec2 uv;
 uniform vec3 right,up,view; uniform vec2 scale;out vec2 texcoord;out float shade;
 void main(){vec3 t=p-vec3(0.,1.14,-.005);gl_Position=vec4(dot(t,right)*scale.x,dot(t,up)*scale.y,-dot(t,view)*.24,1.);texcoord=uv;vec3 lamp=normalize(right*(-.40)+up*.58+view*.70);shade=.90+.12*max(0.,dot(n,lamp))+.025*max(0.,dot(n,up));}`;
 const frag=`#version 300 es
 precision highp float;uniform sampler2D atlas;in vec2 texcoord;in float shade;out vec4 color;void main(){color=vec4(texture(atlas,texcoord).rgb*shade,1.);}`;
 const prog=gl.createProgram();for(const [type,src] of [[gl.VERTEX_SHADER,vert],[gl.FRAGMENT_SHADER,frag]]){const sh=gl.createShader(type);gl.shaderSource(sh,src);gl.compileShader(sh);if(!gl.getShaderParameter(sh,gl.COMPILE_STATUS))throw Error(gl.getShaderInfoLog(sh));gl.attachShader(prog,sh);}
 gl.linkProgram(prog);if(!gl.getProgramParameter(prog,gl.LINK_STATUS))throw Error('渲染程序未能启动');gl.useProgram(prog);
 const loc=Object.fromEntries(['right','up','view','scale'].map(k=>[k,gl.getUniformLocation(prog,k)]));
 const b=Uint8Array.from(atob(document.querySelector('#asset').textContent.trim()),c=>c.charCodeAt(0));const d=new DataView(b.buffer);const jl=d.getUint32(12,true),doc=JSON.parse(new TextDecoder().decode(b.subarray(20,20+jl))),off=28+jl;
 const bytes=i=>{const v=doc.bufferViews[i];return b.subarray(off+(v.byteOffset||0),off+(v.byteOffset||0)+v.byteLength)};
 const img=new Image(),url=URL.createObjectURL(new Blob([bytes(doc.images[0].bufferView)],{type:'image/png'}));img.src=url;await img.decode();
 const texture=gl.createTexture();gl.bindTexture(gl.TEXTURE_2D,texture);gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA,gl.RGBA,gl.UNSIGNED_BYTE,img);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.LINEAR_MIPMAP_LINEAR);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.LINEAR);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);gl.generateMipmap(gl.TEXTURE_2D);URL.revokeObjectURL(url);
 function buffer(target,data){let x=gl.createBuffer();gl.bindBuffer(target,x);gl.bufferData(target,data,gl.STATIC_DRAW)}
 const draws=[];for(const m of doc.meshes){const p=m.primitives[0],vao=gl.createVertexArray();gl.bindVertexArray(vao);for(const [i,k] of ['POSITION','NORMAL','TEXCOORD_0'].entries()){let a=doc.accessors[p.attributes[k]];buffer(gl.ARRAY_BUFFER,bytes(a.bufferView));gl.enableVertexAttribArray(i);gl.vertexAttribPointer(i,i===2?2:3,gl.FLOAT,false,0,0);}const ind=doc.accessors[p.indices];buffer(gl.ELEMENT_ARRAY_BUFFER,bytes(ind.bufferView));draws.push([vao,ind.count,ind.componentType]);}
 gl.enable(gl.DEPTH_TEST);gl.clearColor(247/255,245/255,242/255,1);
 let a=.20,e=.025,zoom=1,spin=false,prev=0,pending=false;const touches=new Map();const rad=Math.PI/180;const spinButton=document.querySelector('#spin');
 function active(){document.querySelectorAll('[data-a]').forEach(el=>el.setAttribute('aria-pressed',String(Math.abs(Math.atan2(Math.sin(a-Number(el.dataset.a)*rad),Math.cos(a-Number(el.dataset.a)*rad)))<.015&&Math.abs(e)<.01)));spinButton.setAttribute('aria-pressed',String(spin));}
 function queue(){if(!pending){pending=true;requestAnimationFrame(draw)}}
 function draw(t){pending=false;if(spin&&prev)a+=Math.min((t-prev)*.00028,.02);prev=t;
 const s=Math.min(devicePixelRatio||1,2),w=Math.max(1,Math.round(canvas.clientWidth*s)),h=Math.max(1,Math.round(canvas.clientHeight*s));if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h}
 const sn=Math.sin(a),cs=Math.cos(a),se=Math.sin(e),ce=Math.cos(e);const k=Math.min(.80,w/h*1.03)*zoom;
 gl.viewport(0,0,w,h);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);gl.uniform3f(loc.right,cs,0,-sn);gl.uniform3f(loc.up,-sn*se,ce,-cs*se);gl.uniform3f(loc.view,sn*ce,se,cs*ce);gl.uniform2f(loc.scale,k*h/w,k);
 for(const [vao,count,type] of draws){gl.bindVertexArray(vao);gl.drawElements(gl.TRIANGLES,count,type,0)}active();if(spin)queue();}
 const limit=()=>{zoom=Math.min(3.5,Math.max(.55,zoom));e=Math.min(1.46,Math.max(-1.46,e));queue()};
 document.querySelectorAll('[data-a]').forEach(el=>el.onclick=()=>{a=Number(el.dataset.a)*rad;e=0;spin=false;queue()});document.querySelector('#reset').onclick=()=>{a=.20;e=.025;zoom=1;spin=false;queue()};spinButton.onclick=()=>{spin=!spin;prev=0;queue()};
 canvas.onpointerdown=ev=>{canvas.setPointerCapture(ev.pointerId);touches.set(ev.pointerId,[ev.clientX,ev.clientY]);spin=false;queue()};
 canvas.onpointermove=ev=>{if(!touches.has(ev.pointerId))return;const old=touches.get(ev.pointerId),other=[...touches.keys()].find(k=>k!==ev.pointerId);if(other!==undefined){const q=touches.get(other);zoom*=Math.hypot(ev.clientX-q[0],ev.clientY-q[1])/Math.max(1,Math.hypot(old[0]-q[0],old[1]-q[1]));}else{a-=(ev.clientX-old[0])*.008;e+=(ev.clientY-old[1])*.006}touches.set(ev.pointerId,[ev.clientX,ev.clientY]);limit()};
 canvas.onpointerup=canvas.onpointercancel=ev=>touches.delete(ev.pointerId);
 canvas.addEventListener('wheel',ev=>{ev.preventDefault();zoom*=Math.exp(-ev.deltaY*.0013);limit()},{passive:false});
 canvas.onkeydown=ev=>{if(!['ArrowLeft','ArrowRight','ArrowDown','ArrowUp','+','-','='].includes(ev.key))return;ev.preventDefault();spin=false;if(ev.key==='ArrowLeft')a-=.12;if(ev.key==='ArrowRight')a+=.12;if(ev.key==='ArrowDown')e-=.10;if(ev.key==='ArrowUp')e+=.10;if(ev.key==='+'||ev.key==='=')zoom*=1.12;if(ev.key==='-')zoom/=1.12;limit()};
 new ResizeObserver(queue).observe(canvas);fallback.hidden=true;status.textContent=matchMedia('(pointer:coarse)').matches?'拖动旋转 · 双指缩放':'拖动旋转 · 滚轮缩放';queue()
 }catch(err){canvas.hidden=true;status.textContent=err.message;document.querySelector('footer').hidden=true;const video=document.createElement('video');video.src='data:video/mp4;base64,@@MOVIE@@';video.controls=true;video.loop=true;video.playsInline=true;video.poster=fallback.src;video.setAttribute('aria-label','角色模型 360 度旋转展示');fallback.parentNode.appendChild(video);fallback.hidden=true;status.textContent='当前显示 360° 旋转视频';console.error(err)}
})();
</script></html>"""
enc=lambda b:base64.b64encode(b).decode()
img=Image.open(R/'制作'/'视角检查.png')
img.thumbnail((620,725));stream=io.BytesIO();img.save(stream,'JPEG',quality=87)
(R/'旋转查看模型.html').write_text(html.replace('@@MODEL@@',enc((R/'蓝鲸女仆.glb').read_bytes())).replace('@@MOVIE@@',enc((R/'模型旋转展示.mp4').read_bytes())).replace('@@POSTER@@','data:image/jpeg;base64,'+enc(stream.getvalue())),'utf-8')
