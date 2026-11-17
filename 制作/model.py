from pathlib import Path
import math, json, struct, io, re
import numpy as np
from PIL import Image
import cv2
from scipy.ndimage import distance_transform_edt, map_coordinates
from scipy.interpolate import PchipInterpolator, CubicSpline
from scipy.spatial import Delaunay
ROOT=Path(__file__).resolve().parents[1]
REF=Path(__file__).with_name('三视图原稿.jpg')
SRC=np.array(Image.open(REF).convert('RGB'))
PI=np.pi
def rgb(c): return np.array(c,dtype=float)
def mix(a,b,t): return a+(b-a)*np.asarray(t)[...,None]
def ss(a,b,t):
    q=np.clip((t-a)/(b-a),0,1);return q*q*(3-2*q)
def interp(x,tab):
    tab=np.array(tab,float);return PchipInterpolator(tab[:,0],tab[:,1:],axis=0)(np.clip(x,tab[0,0],tab[-1,0]))
def curve(p,n,closed=False):
    p=np.array(p,float)
    if closed:p=np.vstack([p,p[0]])
    ts=np.r_[0,np.cumsum(np.maximum(np.linalg.norm(np.diff(p[:,:min(p.shape[1],3)],axis=0),axis=1),.001)**.65)]
    t=np.linspace(0,ts[-1],n,endpoint=not closed)
    return CubicSpline(ts,p,bc_type='periodic' if closed else 'natural')(t)
def sample(im,x,y):
    x,y=np.broadcast_arrays(x,y);shape=x.shape
    return cv2.remap(im,np.ascontiguousarray(x.astype('f').reshape(-1,1)) if x.size<32760 else np.ascontiguousarray(x.astype('f')),np.ascontiguousarray(y.astype('f').reshape(-1,1)) if x.size<32760 else np.ascontiguousarray(y.astype('f')),cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE).reshape(*shape,3).astype(float)
def poly_mask(points):
    m=np.zeros(SRC.shape[:2],np.uint8);cv2.fillPoly(m,[np.array(points,np.int32)],1);return m>0
def filled(points,valid=None,holes=(),erode=0):
    poly=poly_mask(points)
    if valid is not None: poly &= valid
    for h in holes:poly &= ~poly_mask(h)
    if erode:poly=cv2.erode(poly.astype('uint8'),np.ones((erode,erode),'uint8'))>0
    ix=distance_transform_edt(~poly,return_distances=False,return_indices=True)
    return SRC[ix[0],ix[1]]
R,G,B=SRC.transpose(2,0,1).astype(float)
hair_sel=(B>R+13)&(B>G+10)&(R<174)&(G<184)
head_front=[(204,305),(210,232),(252,163),(305,104),(392,82),(477,91),(541,130),(563,223),(550,268),(568,319),(554,360),(536,389),(480,392),(465,370),(496,344),(500,303),(468,272),(444,227),(437,266),(418,310),(424,325),(382,327),(355,300),(339,243),(333,235),(311,288),(300,308),(302,352),(323,370),(317,387),(263,388),(222,363)]
hf=filled(head_front,hair_sel)
back_pts=[(1265,235),(1274,190),(1310,122),(1360,96),(1423,83),(1486,99),(1545,145),(1584,231),(1588,315),(1591,362),(1623,429),(1630,483),(1658,495),(1661,552),(1621,563),(1578,565),(1534,596),(1469,602),(1385,606),(1284,597),(1201,570),(1206,481),(1233,421),(1257,372),(1266,310)]
hb=filled(back_pts,hair_sel,holes=[[(1480,585),(1500,561),(1501,533),(1522,505),(1534,502),(1552,534),(1560,562),(1615,554),(1634,565),(1622,590),(1566,623),(1484,635)],[(1228,222),(1300,219),(1290,265),(1256,318),(1200,312)]],erode=2)
side_pts=[(720,305),(719,231),(740,160),(805,103),(852,96),(901,108),(945,83),(1001,82),(1062,111),(1090,154),(1117,224),(1115,315),(1109,359),(1132,415),(1147,458),(1140,505),(1150,520),(1130,547),(1108,553),(1090,579),(1041,596),(970,594),(938,556),(910,520),(903,437),(922,378),(914,366),(891,392),(850,429),(831,451),(822,438),(839,406),(841,373),(822,371),(839,340),(848,307),(809,251),(802,230),(783,321),(768,326),(736,323)]
hs=filled(side_pts,hair_sel,holes=[[(847,85),(932,65),(978,112),(986,223),(969,243),(988,263),(1036,294),(1047,318),(988,343),(924,324),(904,292),(905,247),(935,243),(918,191),(875,145)],[(910,334),(939,334),(948,365),(929,383),(909,380)]],erode=2)
def safe_area(pts,select,holes):
    poly=poly_mask(pts)&select
    for h in holes:poly &= ~poly_mask(h)
    return cv2.distanceTransform(poly.astype('uint8'),cv2.DIST_L2,5)
hholes=[[(544,217),(608,214),(613,288),(545,284)]]
backholes=[[(1470,590),(1496,554),(1498,527),(1514,501),(1539,496),(1558,530),(1564,555),(1636,550),(1642,582),(1602,622),(1481,639)],[(1230,214),(1302,214),(1295,269),(1252,325),(1180,321)],[(1570,213),(1620,212),(1630,324),(1570,329)]]
sholes=[[(838,73),(936,52),(984,110),(986,221),(999,270),(1052,301),(1049,329),(968,353),(915,329),(897,294),(900,234),(925,229),(910,179)],[(909,331),(942,327),(951,366),(928,389),(906,381)],[(920,484),(1000,528),(997,595),(928,580)]]
# Fade painted linework only where the source contains the corresponding surface.
# Excluded accessories are replaced by a continuous hair color field.
validHair=(B>R+9)&(B>G+7)&(R<180)&(G<183)
fm=safe_area(head_front,validHair,hholes)
bm=safe_area(back_pts,validHair,backholes)
sm=safe_area(side_pts,validHair,sholes)
def srcweight(mask,x,z,k):
    return np.clip(cv2.remap(mask,x.astype('f'),z.astype('f'),cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)/k,0,1)
def hair(p,u,v,force=None):
    x,y,z=p
    base=mix(rgb([79,91,159]),rgb([107,135,200]),ss(311,586,z))
    if force=='capeFront':return mix(rgb([72,82,145]),rgb([106,140,207]),ss(321,568,z))
    if force=='fringe':
        w=srcweight(fm,x,z,3.6)
        return mix(base,sample(SRC,x,z),w)
    # Back painting is sampled entirely within the hair mass.
    bx=1421-(x-403)*.93
    sb=sample(SRC,bx,z);wb=srcweight(bm,bx,z,8)
    # The tiara has its own sculpted geometry.
    high=mix(base,np.maximum(base,sb),ss(80,125,sb[...,0]))
    sb=mix(high,sb,ss(238,298,z))
    side=sample(SRC,y,z);ws=srcweight(sm,y,z,9)
    wave=ss(325,575,z)*(.5+.5*np.cos((x-403)*.087+(y-926)*.031+z*.014))
    c=base*(1-.065*wave)[...,None]
    # Small painted crown glints on the outer temples.
    gl=np.exp(-((z-165)/10)**2-((y-1047)/18)**2)+.6*np.exp(-((z-181)/6)**2-((y-1031)/8)**2)
    c=mix(c,rgb([143,154,213]),np.clip(gl*.55,0,1))
    c=mix(c,sb,wb*ss(949,1028,y)*(1-ss(480,530,z)*ss(1465,1492,bx)))
    return c
SKIN=rgb([255,233,223]);WHITE=rgb([252,244,242]);NAVY=rgb([49,47,83]);GOLD=rgb([217,178,126]);INK=rgb([49,39,62])
def face_tex():
    yy,xx=np.indices(SRC.shape[:2])
    c=np.tile(SKIN,(*SRC.shape[:2],1))
    blush=(np.exp(-((xx-334)/29)**2-((yy-359)/12)**2)+np.exp(-((xx-471)/29)**2-((yy-359)/12)**2))*.60
    c=mix(c,rgb([255,190,182]),blush)
    for pts in [[(294,306),(303,291),(317,285),(342,286),(358,293),(365,307),(365,332),(357,352),(340,355),(319,349),(310,336),(307,316)],[(427,301),(437,289),(446,284),(478,284),(489,289),(506,303),(501,319),(493,326),(487,346),(477,353),(451,356),(437,348),(429,335)],[(324,281),(348,279),(354,287),(327,288)],[(438,278),(471,277),(477,285),(439,287)],[(392,367),(395,364),(399,365),(402,369),(399,371),(393,370)]]:
        mask=poly_mask(pts).astype('f'); mask=cv2.GaussianBlur(mask,(3,3),.6);c=mix(c,SRC.astype(float),mask)
    return c.astype('uint8')
ft=face_tex()
def face(p,u,v):
    x,y,z=p
    return mix(sample(ft,x,z),np.broadcast_to(SKIN,x.shape+(3,)),ss(1.04,1.49,abs(u-.5)*2*PI))
MESH=[];group='Body'
def add(name,pts,faces,uv,tex,mat,weld=True):
    global group
    MESH.append(dict(name=name,group=group,p=np.array(pts,dtype='f').reshape(-1,3),f=np.array(faces,dtype=np.int32).reshape(-1,3),uv=np.array(uv,'f').reshape(-1,2),tex=tex,mat=mat,weld=weld))
def Q(x,y,z):return np.stack(np.broadcast_arrays(x,y,z),axis=-1)
def grid(nu,nv):
    return np.meshgrid(np.linspace(0,1,nu),np.linspace(0,1,nv))
def rect(name,func,paint,nu=56,nv=38,res=(200,180),mat='Cloth',flip=False):
    u,v=grid(nu,nv);p=func(u,v)
    q=np.arange(nu*(nv-1)).reshape(nv-1,nu)[:,:-1].ravel()
    f=np.vstack([np.array([q,q+1,q+nu]).T,np.array([q+1,q+nu+1,q+nu]).T])
    if flip:f=f[:,::-1]
    if callable(paint):
        a,b=grid(*res); t=func(a,b).transpose(2,0,1); col=paint(t,a,b)
        tex=Image.fromarray(np.uint8(np.clip(col,0,255)))
    else:tex=tuple(paint)
    add(name,p,f,Q(u,v,0)[...,:2],tex,mat)
def loft(name,tab,paint,fold=0,pleats=12,mat='Cloth',res=(360,220),nu=100,nv=42,edge=0,hem=0):
    # z, center x, front, back, radius in x
    t=np.array(tab,float)
    if name.startswith('Face ·'):globals()['head_sections']=t
    def geo(u,v):
        z=t[0,0]+(t[-1,0]-t[0,0])*v
        cx,fr,bk,rx=interp(z,t).transpose(2,0,1)
        a=(u-.5)*2*PI; wob=1+fold*np.cos(a*pleats+.24*np.sin(3*a))*v**1.4
        x=cx+rx*np.sin(a)*wob;y=(fr+bk)/2-(bk-fr)/2*np.cos(a)*wob
        if name.startswith('Face ·'):
            target=fr+interp(np.abs(x-cx),[(0,0),(14,7),(35,23),(60,38),(85,55),(99,70),(116,114),(136,135)])[...,0]
            weight=(1-ss(.99,1.51,abs(a)))*ss(195,247,z)*(1-ss(363,393,z))
            y=y*(1-weight)+target*weight
        return Q(x,y,z-edge*(1-np.cos(a))*v/2+hem*np.cos(pleats*a+.3)*v)
    rect(name,geo,paint,nu,nv,res,mat,flip=True)
def ball(name,center,rad,paint,mat='Detail',res=(96,64),tilt=0):
    def geo(u,v):
        a=2*PI*u; h=(v-.5)*PI; x=rad[0]*np.cos(h)*np.sin(a);z=rad[2]*np.sin(h)
        return Q(center[0]+x*np.cos(tilt)-z*np.sin(tilt),center[1]-rad[1]*np.cos(h)*np.cos(a),center[2]+x*np.sin(tilt)+z*np.cos(tilt))
    rect(name,geo,paint,37,23,res,mat,flip=True)
def sweep(name,path,paint,mat='Detail',res=(130,190),nu=21,nv=68,pref=(0,1,0)):
    # center, width, thickness. First profile axis across path, second facing pref.
    path=np.array(path,float);ts=np.r_[0,np.cumsum(np.maximum(np.linalg.norm(np.diff(path[:,:3],axis=0),axis=1),.1)**.7)];ts/=ts[-1]
    spline=CubicSpline(ts,path[:,:3],bc_type='natural'); radii=PchipInterpolator(ts,path[:,3:])
    def geo(u,v):
        c=spline(v);t=spline(v,1);t/=np.maximum(np.linalg.norm(t,axis=-1,keepdims=True),1e-8)
        w=np.cross(t,np.broadcast_to(pref,t.shape));w/=np.maximum(np.linalg.norm(w,axis=-1,keepdims=True),1e-9)
        h=np.cross(w,t); r=radii(v);a=u*2*PI
        return c+w*(np.cos(a)*r[...,0])[...,None]+h*(np.sin(a)*r[...,1])[...,None]
    rect(name,geo,paint,nu,nv,res,mat)
def pad(name,outline,depth,paint,mat='Detail',thick=4,crown=3,step=6,res=None):
    # A closed, domed volume with its own curved front, rolled lip and back.
    boundary=curve(outline,max(65,len(outline)*6),True)
    lo=boundary.min(axis=0)-1;hi=boundary.max(axis=0)+1
    W,H=np.ceil(hi-lo).astype(int)+2
    mask=np.zeros((H,W),np.uint8);cv2.fillPoly(mask,[np.round(boundary-lo).astype('i')],1)
    dst=cv2.distanceTransform(mask,cv2.DIST_L2,5)
    yy,xx=np.mgrid[step/2:H:step,step/2:W:step]; ins=dst[yy.astype(int),xx.astype(int)]>step*.55
    xy=np.vstack((boundary,np.c_[xx[ins],yy[ins]]+lo));n=len(xy);nb=len(boundary)
    tri=Delaunay(xy).simplices; mids=xy[tri].mean(axis=1)-lo
    good=mask[np.clip(np.rint(mids[:,1]).astype(int),0,H-1),np.clip(np.rint(mids[:,0]).astype(int),0,W-1)]>0
    tri=tri[good]
    x,z=xy.T;uv=(xy-lo)/(hi-lo)
    def ds(x,z):return map_coordinates(dst,[np.clip(z-lo[1],0,H-1),np.clip(x-lo[0],0,W-1)],order=1)
    def depth_fn(x,z):
        d=ds(x,z); return depth(x,z)-crown*np.sin(np.pi/2*np.clip(d/max(4,min(W,H)*.23),0,1))
    y=depth_fn(x,z);pt=np.vstack((Q(x,y,z),Q(x,depth(x,z)+thick,z)))
    i=np.arange(nb);j=(i+1)%nb
    side=np.vstack((np.c_[i,j,j+n],np.c_[i,j+n,i+n]))
    # Imported geometry maps x/right, z/down to world X/Y. Front wound clockwise in image.
    front=tri[:,::-1];back=tri+n
    area=(boundary[:,0]*np.roll(boundary[:,1],-1)-boundary[:,1]*np.roll(boundary[:,0],-1)).sum()
    if area>0:side=side[:,::-1]
    if callable(paint):
        w,h=res or (max(16,min(648,int(W*1.7))),max(16,min(740,int(H*1.7))))
        a,b=grid(w,h);tx=lo[0]+a*(hi[0]-lo[0]);tz=lo[1]+b*(hi[1]-lo[1]);ty=depth_fn(tx,tz)
        tex=Image.fromarray(np.uint8(np.clip(paint((tx,ty,tz),a,b),0,255)))
    else:tex=tuple(paint)
    add(name,pt,np.vstack((front,back,side)),np.vstack((uv,uv)),tex,mat)
def const(n):return lambda x,z:np.full_like(x,n)
def proj(im):return lambda p,u,v:sample(im,p[0],p[2])
def cliptex(pts,valid=None,holes=()):return filled(pts,valid,holes)
group='01 · Face and body'
loft('Face · painted porcelain',[(158,402,840,992,8),(186,402,798,1023,76),(221,402,772,1040,115),(258,402,766,1047,123),(291,402,768,1041,119),(321,402,768,1028,117),(343,402,761,1015,114),(357,402,758,1003,108),(370,402,765,987,108),(381,402,780,969,86),(390,402,804,947,57),(395,402,836,920,22),(396,402,864,885,1)],face,mat='Porcelain',res=(680,440),nu=145,nv=80)
ball('Neck',(401,876,398),(23,25,30),SKIN,'Porcelain')
for k in [-1,1]:
    ball('Ear '+str(k),(402+k*133,917,348),(16,22,24),[253,221,211],'Porcelain')
    ball('Ear hollow '+str(k),(402+k*145,914,348),(4,12,15),[237,188,182],'Porcelain')

group='02 · Hair'
shelltab=[(79,1,938,941),(81,33,905,971),(87,63,872,1003),(104,102,825,1037),(140,147,775,1077),(181,176,746,1100),(222,193,727,1110),(265,202,720,1116),(308,202,732,1111),(350,192,754,1104),(393,190,780,1108),(440,194,798,1124),(494,210,812,1131),(531,214,821,1135),(572,194,823,1110),(605,152,837,1068)]
def cap_info(z):return interp(z,shelltab)
def frontdepth(x,z):
    rx,fr,bk=np.moveaxis(cap_info(z),-1,0)
    return (fr+bk)/2-(bk-fr)/2*np.sqrt(np.maximum(.018,1-((x-403)/np.maximum(rx,1))**2))
endtab=[(0,170),(.38,181),(.66,220),(.84,232),(1.03,274),(1.19,352),(1.26,395),(1.29,480),(1.35,541),(1.44,563),(1.56,574),(1.86,565),(2.08,578),(2.4,587),(2.68,580),(2.93,591),(PI,583)]
def shell_geo(u,v,inner=False):
    a=(u-.5)*2*PI; ab=abs(a)
    end=interp(ab,endtab)[...,0]+(6*np.sin(12*a)+5*np.cos(7*a))*ss(.99,1.49,ab)
    t=v**1.65;z=79+(end-79)*t
    rx,fr,bk=np.moveaxis(cap_info(z),-1,0)
    flute=ss(318,585,z)*(5*np.cos(11*a+(z-390)*.012)+3*np.sin(7*a))
    xx=403+(rx*(1-.085*(1-ss(349,481,z))-.11*ss(378,508,z)*np.sin(a)**4)+flute)*np.sin(a); yy=(fr+bk)/2-((bk-fr)/2+flute)*np.cos(a)
    # The crown follows the same facial curvature as the separate fringe.
    # Its front surface remains under the parted locks.
    dx=np.abs(xx-403)
    brow=fr+dx*1.04*ss(0,61,dx)
    frontplane=frontdepth(xx,z)*(1-ss(118,265,z))+brow*ss(118,265,z)
    follow=(1-ss(1.27,1.68,ab))*ss(80,118,z)*(1-ss(350,396,z))
    yy+=np.maximum(0,frontplane+19-yy)*follow
    if inner:
        inz=ss(90,290,z)
        xx=403+(xx-403)*(1-.15*inz); yy=924+(yy-924)*(1-.17*inz);z=z-5*inz
    return Q(xx,yy,z)
rect('Crown and flowing back hair',shell_geo,hair,221,96,(960,660),'Indigo hair',flip=True)
rect('Hair lining',lambda u,v:shell_geo(u,v,True),lambda p,u,v:hair(p,u,v)*.84,125,54,(400,300),'Indigo hair')
def hemgeo(u,v):
    o=shell_geo(u,np.ones_like(v));i=shell_geo(u,np.ones_like(v),True);return mix(o,i,v)
rect('Rounded hem of back hair',hemgeo,lambda p,u,v:hair(p,u,v)*.91,201,5,(420,20),'Indigo hair',flip=True)
fringe=[(208,298),(215,249),(236,207),(257,167),(286,131),(324,104),(359,91),(400,87),(447,92),(491,111),(530,141),(554,178),(569,221),(577,265),(575,312),(564,345),(549,371),(531,384),(537,368),(520,380),(482,388),(496,375),(475,374),(494,360),(504,340),(506,306),(489,291),(462,264),(441,223),(438,247),(431,277),(415,309),(424,323),(406,322),(390,326),(393,317),(379,311),(382,322),(359,302),(347,275),(337,231),(322,267),(305,292),(288,308),(289,335),(301,354),(316,361),(306,362),(318,375),(304,375),(322,385),(282,380),(264,371),(272,385),(246,371),(231,350),(224,315),(227,339),(217,325)]
def bangdepth(x,z):
    dx=abs(x-403);rx,fr,bk=np.moveaxis(cap_info(z),-1,0)
    side=fr+dx*1.04*ss(0,61,dx)
    surface=frontdepth(x,z)*(1-ss(118,265,z))+side*ss(118,265,z)-3-20*ss(280,355,z)*ss(65,105,dx)
    cx,fr,bk,rx=np.moveaxis(interp(z,head_sections),-1,0)
    ang=np.arcsin(np.clip(abs(x-cx)/rx,0,1))
    target=fr+interp(abs(x-cx),[(0,0),(14,7),(35,23),(60,38),(85,55),(99,70),(116,114),(136,135)])[...,0]
    weight=(1-ss(.99,1.51,ang))*ss(195,247,z)*(1-ss(363,393,z))
    skin=((fr+bk)/2-(bk-fr)/2*np.cos(ang))*(1-weight)+target*weight
    return np.minimum(surface,skin-6+200*(1-ss(205,267,z))+np.maximum(0,abs(x-cx)-rx)*2)
pad('Bangs · sculpted parted fringe',fringe,bangdepth,lambda p,u,v:hair(p,u,v,'fringe'),'Indigo hair',thick=8,crown=4,step=5)
# Separated, curved locks. Each has a rounded lenticular cross-section.
frontcurl=[(258,977,331,.6,.6),(264,966,352,17,12),(274,936,377,25,18),(290,889,397,23,16),(298,861,414,19,14),(293,846,433,13,10),(281,841,443,8,5),(271,842,438,.7,.5)]
def lockpaint(p,u,v):
    x,y,z=p
    c=mix(rgb([68,77,138]),rgb([118,157,221]),ss(340,455,z))
    shade=.92+.09*np.sin(2*PI*u)+.055*np.cos(2*PI*u*2)
    c=c*shade[...,None]
    streak=np.exp(-((np.sin(2*PI*(u-.09))+.12*np.sin(PI*v))/.10)**2)*np.sin(PI*v)**1.5
    return mix(c,rgb([60,79,139]),.34*streak)
for k in [1,-1]:
    def mirror(path):return [(403+k*(x-403),y,z,r,d) for x,y,z,r,d in path]
    sweep('Cheek curl '+str(k),mirror(frontcurl),lockpaint,'Indigo hair',res=(96,180))
    for n,path in enumerate([
    [(244,953,292,18,14),(237,944,360,28,21),(240,958,405,28,19),(213,974,457,31,20),(206,981,508,25,16),(185,983,519,14,10),(173,978,502,.6,.4)],
    [(240,975,341,13,12),(223,986,398,20,15),(200,1012,435,16,12),(189,1035,468,11,8),(198,1049,495,7,5),(213,1053,502,.5,.5)],
    [(267,971,365,26,17),(262,986,422,37,22),(239,1001,474,33,23),(240,1016,527,28,21),(221,1018,551,20,14),(199,1008,563,10,7),(186,998,557,.5,.4)],
    [(279,983,382,22,18),(280,1008,446,34,25),(268,1037,502,38,23),(267,1059,546,27,19),(256,1066,565,12,10),(246,1062,572,.4,.4)],
    [(233,951,310,6,4),(222,962,363,6,4),(201,986,403,5,3.5),(199,1003,429,4,3),(208,1004,442,1,1)]
    ]):
        sweep('Surf curl %s.%s'%(k,n+1),mirror(path),lambda p,u,v:hair(p,u,v,'capeFront')*(.96+.095*np.sin(u*2*PI)-.07*np.cos(4*PI*u)-.040*np.cos(8*PI*u+.25*np.sin(PI*v)))[...,None],'Indigo hair',res=(90,200))
# Rear locks turn upwards at the edge of the mantle, visible in true profile.
for k in [-1,1]:
    for n,path in enumerate([
      [(79,1044,317,.5,.5),(108,1060,406,24,15),(113,1085,457,28,17),(113,1104,510,26,18),(107,1104,537,16,12),(105,1124,545,10,7),(100,1144,525,.5,.4)],
      [(116,1022,333,.8,.6),(132,1028,410,19,15),(139,1046,462,27,19),(141,1063,512,30,19),(139,1068,548,23,16),(134,1086,568,10,8),(129,1100,562,.6,.5)],
      [(100,1080,337,.7,.5),(107,1098,393,8,5),(103,1121,429,7,5),(97,1132,458,5,3),(94,1123,483,3,2),(91,1110,488,.4,.3)]
    ]):
        pp=[(403+k*x,y,z,r,d) for x,y,z,r,d in path]
        sweep('Rear cascade %s.%s'%(k,n),pp,lambda p,u,v:hair(p,u,v,'capeFront')*(.96+.072*np.sin(u*2*PI)-.045*np.cos(4*PI*u))[...,None],'Indigo hair',res=(96,180),pref=(1,0,0))
# Ahoge sweeps upwards, then curls to one side.
sweep('Crown cowlick',[(398,870,100,1,1),(401,876,63,3.4,3),(392,868,32,7,5),(372,854,17,7,5),(347,835,18,6.8,5),(321,810,30,7.2,5),(302,790,49,7.2,5),(292,777,69,5,3.4),(296,776,80,.5,.4)],lambda p,u,v:mix(rgb([48,50,98]),rgb([100,115,179]),(.45+.43*np.sin(2*PI*u))),'Indigo hair',res=(76,170))

group='03 · Dress and petticoat'
# Shoes and stockings
for k in [-1,1]:
    cx=402+k*41
    loft('Stocking '+str(k),[(607,cx,857,900,18),(627,cx,852,902,23),(648,cx,850,903,26),(659,cx,850,901,25),(669,cx,854,900,22),(678,cx,864,896,15)],lambda p,u,v:mix(rgb([255,244,241]),rgb([249,221,218]),.46*np.exp(-((p[2]-655)/3)**2)),mat='Cotton',res=(130,100),nu=52,nv=29)
    loft('Mary Jane '+str(k),[(660,cx,857,903,17),(667,cx,849,907,24),(677,cx,840,909,27),(688,cx,838,904,24),(694,cx,844,900,18),(695,cx,861,888,6)],lambda p,u,v:mix(rgb([43,41,74]),rgb([79,76,115]),.46*np.exp(-((p[2]-678)/4)**2)),mat='Leather',res=(150,78),nu=65,nv=24)
    ball('Shoe opening '+str(k),(cx,855,670),(16,12,6),[252,237,233],'Cotton')
    sweep('Instep strap '+str(k),[(cx-23,870,673,1,1),(cx-13,855,666,3,2),(cx,850,665,3,2),(cx+13,855,666,3,2),(cx+23,869,673,1,1)],[65,62,97],'Leather',nv=30)
    ball('Shoe clasp '+str(k),(cx+k*19,855,672),(3.1,2.4,3.1),GOLD,'Gilt')
skirtab=[(466,403,814,912,47),(484,403,806,926,57),(512,403,792,942,77),(539,403,780,959,99),(574,403,762,981,128),(596,403,754,990,148),(612,403,754,996,151),(619,403,759,992,148),(625,403,769,984,141)]
def skgeom(x,z):
    cx,fr,bk,rx=np.moveaxis(interp(z,skirtab),-1,0)
    return (fr+bk)/2-(bk-fr)/2*np.sqrt(np.maximum(.05,1-((x-cx)/rx)**2))
# Extract reference embroidery along its curved hem and repeat it around the skirt.
def textile(p,u,v):
    x,y,z=p;a=(u-.5)*2*PI
    wave=.5+.5*np.cos(12*a+.4*np.sin(3*a)); base=mix(rgb([45,44,77]),rgb([68,65,111]),.18+.74*wave)
    local=(z-625)+10*(1-np.cos(a))/2
    # packed repeating motifs sampled as a flat decorative band, then wrapped around folds
    tx=326+(np.mod(a/(2*PI)*7+.5,1))*146
    srcz=600-.00094*(tx-401)**2 + (local+25)*.86
    dec=sample(SRC,tx,srcz)
    color=np.clip((dec[...,0]-85)/40,0,1)*ss(-42,-35,local)*(1-ss(-13,-10,local))
    base=mix(base,dec,color)
    for dist,width,tint in [(-12,1.5,[211,167,124]),(-7,1.4,[89,85,126]),(-2.2,2.2,[32,30,58])]:
        base=mix(base,rgb(tint),.92*np.exp(-((local-dist)/width)**2))
    return base
loft('Midnight embroidered bell skirt',skirtab,textile,.024,13,'Jacquard',(920,290),169,54,edge=10)
def pett(p,u,v):
    x,y,z=p;a=(u-.5)*2*PI
    r=(np.cos(23*a+.2*np.sin(5*a))+1)/2
    c=mix(rgb([188,173,194]),rgb([254,246,238]),.18+.82*r**.40)
    e=(1-v)*40
    c=mix(c,rgb([193,157,135]),.66*np.exp(-((e-4.8-1.25*np.cos(a*75))/1.)**2))
    flower=np.sqrt((np.sin(a*48)/.63)**2+((e-9.3)/2.25)**2)
    c=mix(c,rgb([211,175,147]),.64*np.exp(-((flower-1)/.36)**2))
    return c
loft('Scalloped ivory petticoat',[(582,403,783,959,119),(597,403,783,966,125),(617,403,772,976,138),(635,403,754,991,154),(651,403,746,998,159)],pett,.032,24,'Lace',(860,165),217,36,edge=14,hem=3)
# discreet fabric lining / underside of skirt
loft('Skirt underside',[(629,403,864,876,2),(632,403,821,925,75),(635,403,779,960,132)], [192,179,195],mat='Lace',nu=89,nv=9)
bod=cliptex([(338,399),(356,389),(379,396),(400,398),(423,394),(459,390),(472,400),(457,452),(450,477),(349,477),(346,453)],holes=[[(328,383),(351,378),(342,399),(337,430),(329,425)]])
def torso(p,u,v):
    a=(u-.5)*2*PI
    return mix(np.broadcast_to(NAVY,p[0].shape+(3,)),sample(bod,p[0],p[2]),1-ss(.68,1.37,abs(a)))
loft('Fitted bodice and bib',[(389,402,845,905,27),(398,402,833,916,47),(414,402,826,926,65),(434,402,818,929,66),(457,402,813,922,55),(478,402,811,917,53)],torso,mat='Jacquard',res=(415,175))
for k in [-1,1]:
    cx=lambda x:403+k*(x-403)
    sleeve=[(cx(459),887,413,5,5),(cx(468),886,432,20,23),(cx(479),883,454,22,27),(cx(493),879,476,18,24),(cx(510),875,498,23,23),(cx(515),874,507,21,21),(cx(517),875,513,15,18)]
    sweep('Puffed sleeve '+str(k),sleeve,lambda p,u,v:mix(rgb([44,43,77]),rgb([68,63,106]),(.45+.20*np.cos(2*PI*u)+.16*np.sin(9*PI*u)*np.sin(v*PI))),'Jacquard',res=(164,150))
    for j,(z,w,col) in enumerate([(505,24,[45,42,76]),(510,24,[102,90,117]),(514,21,[224,193,157]),(520,21,[247,237,231])]):
        sweep('Cuff layer %s.%s'%(k,j),[(cx(514+(z-506)*.36),874,z-3,w*.94,21),(cx(514+(z-506)*.36+1),874,z,w,22),(cx(514+(z-506)*.36+2),874,z+3,w*.93,20)],col,'Lace' if j==3 else 'Jacquard',nv=8)
    ball('Mitten palm '+str(k),(cx(522),873,530),(16,15,13),SKIN,'Porcelain',tilt=k*.49)
    for j in range(3):
        ball('Fingertip %s.%s'%(k,j),(cx(514+j*7),865,534+2*np.sin(j)),(5.5,8,6),SKIN,'Porcelain')
    ball('Thumb '+str(k),(cx(508),862,526),(6,8,7),SKIN,'Porcelain')
group='04 · Apron and ribbons'
loft('Tied apron waistband',[(473,403,810,919,50),(475,403,809,922,54),(481,403,808,923,56),(484,403,808,923,56)],lambda p,u,v:mix(rgb([221,206,215]),rgb([251,240,236]),.30+.70*np.sin(v*PI)) ,mat='Embroidered linen',res=(220,24),nu=89,nv=9)

ap=[(349,474),(366,473),(401,475),(435,473),(452,472),(456,481),(464,487),(465,495),(471,503),(470,513),(475,522),(472,531),(473,541),(467,548),(463,558),(456,560),(450,569),(442,570),(433,578),(424,578),(410,584),(400,583),(389,584),(381,581),(371,580),(366,575),(353,573),(347,567),(338,565),(334,557),(328,553),(329,545),(323,538),(325,530),(321,522),(323,514),(324,505),(329,498),(330,490),(339,481),(341,476)]
aptex=cliptex(ap)
pad('Whale apron · gathered lace edge',ap,lambda x,z:skgeom(x,np.maximum(474,z))-4,proj(aptex),'Embroidered linen',thick=3,crown=3.8,step=3)
for k in [-1,1]:
    right=[(449,397),(465,393),(472,397),(473,406),(468,411),(467,420),(463,424),(463,431),(460,435),(458,448),(450,458),(447,474),(433,475),(436,455),(440,431),(444,410)]
    pts=[(403+k*(x-403),z) for x,z in right]
    tx=cliptex(right)
    pad('Ruffled shoulder strap '+str(k),pts,lambda x,z:829+11*((x-403)/67)**2-8*ss(429,475,z),lambda p,u,v:sample(tx,403+k*(p[0]-403),p[2]),'Lace',thick=4,crown=3.1,step=3)
bow=[(400,404),(412,400),(432,395),(437,399),(436,416),(432,429),(423,426),(425,441),(414,435),(406,422),(399,418),(388,432),(377,441),(378,428),(366,431),(364,418),(365,400),(371,396),(388,400)]
pad('Sailor throat bow',bow,lambda x,z:828+((x-400)/36)**2*7,proj(cliptex(bow)),'Silk',crown=4,step=2.8)
ball('Silk bow centre',(400,832,410),(11,12.8,9),[52,48,82],'Silk')
ball('Cameo setting',(400,820,410),(9,3.3,10),GOLD,'Gilt')
ball('Ocean cabochon',(400,816.8,410),(6.2,2.2,7.1),[74,130,178],'Enamel')
ball('Cameo reflection',(398,814.6,407),(1.9,.5,2.2),[212,237,246],'Enamel')
for k in [-1,1]:
    pts0=[(474,578),(466,572),(462,571),(462,585),(467,591),(473,586),(469,603),(468,617),(477,611),(481,617),(481,598),(486,616),(492,611),(496,613),(489,588),(497,590),(498,576),(495,574),(483,577),(479,570)]
    pts=[(400+k*(x-400),z) for x,z in pts0]
    tx=cliptex(pts0)
    pad('Skirt ribbon '+str(k),pts,lambda x,z:skgeom(x,z)-4.8,lambda p,u,v:sample(tx,400+k*(p[0]-400),p[2]),'Silk',thick=2.4,crown=2.3,step=2.5)
group='05 · Lace headdress and fins'
def arch(a):
    return Q(401+181*np.sin(a),936-66*np.cos(a),252-166*np.cos(a))
def headdress(u,v):
    a=(u-.5)*PI*.97
    edge=np.sin(PI*u)**.38
    repeats=11
    lobe=(1-np.cos(2*PI*repeats*u))/2
    h=v*(15+edge*(15+11*lobe))
    base=arch(a);base[...,0]+=np.sin(a)*h;base[...,2]-=np.cos(a)*h
    base[...,1]+=v*34+edge*(4*np.sin(2*PI*repeats*u)+3*np.sin(2*PI*repeats*2*u))*v+3*np.sin(v*PI)
    return base
def lacepaint(p,u,v):
    lobe=(np.cos(2*PI*11*u)+1)/2
    col=mix(rgb([251,243,245]),rgb([217,198,218]),.27*lobe+.18*(1-v))
    cx=((u*11)%1)-.5
    e=np.sqrt((cx/.105)**2+((v-.16)/.115)**2)
    col=mix(col,rgb([180,161,191]),.88*np.exp(-((e-1)/.38)**2))
    curl=np.abs(v-(.76+.048*np.cos(2*PI*u*55)+.024*np.cos(2*PI*u*99)))
    col=mix(col,rgb([192,173,202]),.66*np.exp(-(curl/.032)**2))
    col=mix(col,rgb([112,96,124]),.7*np.exp(-((1-v)/.046)**2))
    return col
rect('Pleated lace headdress',headdress,lacepaint,287,18,(820,95),'Lace')
rect('Lace facing',lambda u,v:headdress(u,v)+np.array([0,3,1]),lacepaint,287,16,(460,70),'Lace',flip=True)
rect('Lace scallop piping',lambda u,v:headdress(u,np.ones_like(u))+Q(0,3*v,v),[166,146,167],287,3,mat='Lace')
def bandgeo(u,v):
    a=(u-.5)*PI; base=arch(a);h=(v-.5)*12
    base[...,0]+=np.sin(a)*h;base[...,2]-=np.cos(a)*h;base[...,1]-=5*np.sin(v*PI)+2
    return base
rect('Indigo headband',bandgeo,lambda p,u,v:mix(rgb([46,44,82]),rgb([89,93,146]),np.sin(PI*v)*.69),161,13,(390,42),'Silk')
def rearband(u,v):
    a=(u-.5)*PI
    h=v*9
    return Q(402+(185+h)*np.sin(a),945+21*np.cos(a)-2.5*np.sin(PI*v),253-(171+h)*np.cos(a))
rect('Satin binding behind lace',rearband,lambda p,u,v:mix(rgb([53,53,95]),rgb([83,88,147]),np.sin(PI*v)*.61),161,11,(340,36),'Silk',flip=True)
finpts=[(555,264),(568,254),(582,255),(599,268),(615,280),(635,294),(654,303),(666,310),(662,317),(650,322),(641,324),(631,330),(620,330),(610,335),(600,334),(592,337),(582,334),(574,331),(568,322),(565,299)]
fintex=cliptex(finpts)
bowpts=[(553,227),(560,225),(570,230),(579,239),(584,237),(595,228),(603,226),(606,233),(607,252),(606,263),(599,265),(607,273),(597,276),(584,261),(578,258),(570,275),(559,277),(561,267),(552,265),(551,251)]
bowtex=cliptex(bowpts)
for k in [-1,1]:
    # Wing root lies in the temple; distal feathers point backwards in profile.
    pts=[(403+k*(x-403),z) for x,z in finpts]
    pad('Wing-shaped ear fin '+str(k),pts,lambda x,z:916+(.88*(abs(x-403)-151))+(z-266)*.33,lambda p,u,v:sample(fintex,403+k*(p[0]-403),p[2]),'Feather satin',thick=10,crown=5.4,step=3.2)
    pts=[(403+k*(x-403+7),z) for x,z in bowpts]
    pad('Temple bow '+str(k),pts,lambda x,z:889+1.46*(abs(x-403)-182),lambda p,u,v:sample(bowtex,403+k*(p[0]-403)-7,p[2]),'Silk',thick=11,crown=5.8,step=2.6)
    ball('Temple gold pearl '+str(k),(403+k*184,885,247),(5.7,5,7),[224,184,128],'Gilt')
group='06 · Whale tail'
sweep('Whale tail · peduncle',[(438,945,552,13,15),(459,981,570,17,18),(489,1013,584,19,18),(528,983,579,20,17),(552,929,566,14,13),(575,920,550,5,7)],lambda p,u,v:mix(rgb([53,61,113]),rgb([85,111,168]),.50+.30*np.cos(PI*2*u)),'Ocean satin')
tailpts=[(549,573),(552,560),(555,549),(555,532),(560,516),(570,501),(579,491),(583,490),(587,505),(596,519),(599,532),(598,544),(608,543),(623,548),(635,553),(645,554),(644,561),(637,570),(625,578),(612,584),(595,587),(580,585),(569,582),(556,586),(548,586)]
tailtx=cliptex(tailpts)
pad('Whale tail · twin flukes',tailpts,lambda x,z:898+(x-556)*1.00+(z-548)*.14,proj(tailtx),'Ocean satin',thick=9,crown=6,step=3)
def export():
    tiles=[]; lookup={}
    for m in MESH:
        if isinstance(m['tex'],Image.Image): im=m['tex']
        else:
            key=tuple(np.uint8(m['tex']))
            if key in lookup:m['tile']=lookup[key];continue
            im=Image.new('RGB',(8,8),key)
            lookup[key]=len(tiles)
        m['tile']=len(tiles);tiles.append(im)
    width=2048;pos=[None]*len(tiles);px=py=row=0;pad=4
    for i in sorted(range(len(tiles)),key=lambda i:-tiles[i].height):
        w,h=tiles[i].size
        if px+w+pad*2>width:py+=row;px=row=0
        pos[i]=(px+pad,py+pad);px+=w+2*pad;row=max(row,h+2*pad)
    height=2**math.ceil(math.log2(py+row));atlas=Image.new('RGB',(width,height))
    for im,(x,y) in zip(tiles,pos):
        border=np.pad(np.array(im),((pad,pad),(pad,pad),(0,0)),mode='edge');atlas.paste(Image.fromarray(border),(x-pad,y-pad))
    binbuf=bytearray();acc=[];views=[];nodes=[];meshes=[];materials=[];roots=[];groups={}; md={}
    def buf(data,target=None):
        while len(binbuf)%4:binbuf.append(0)
        v={'buffer':0,'byteOffset':len(binbuf),'byteLength':len(data)}
        if target:v['target']=target
        views.append(v);binbuf.extend(data);return len(views)-1
    def accessor(data,typ,ct,target,box=False):
        v={'bufferView':buf(data.tobytes(),target),'componentType':ct,'count':len(data),'type':typ}
        if box:v.update(min=data.min(0).tolist(),max=data.max(0).tolist())
        acc.append(v);return len(acc)-1
    for m in MESH:
        grp=m['group']; mat=m['mat']
        if grp not in groups:
            groups[grp]=len(nodes);roots.append(len(nodes));nodes.append({'name':grp,'children':[]})
        if mat not in md:
            md[mat]=len(materials)
            materials.append({'name':mat,'doubleSided':True,'pbrMetallicRoughness':{'baseColorTexture':{'index':0},'metallicFactor':0.0,'roughnessFactor':.84 if mat not in ['Gilt','Enamel','Leather'] else .55}})
        p=m['p'];p=(p-np.array([403,920,695]))[:,[0,2,1]]*np.array([1,-1,-1])/300.
        p=p.astype('<f4');f=m['f'];dv=np.cross(p[f[:,1]]-p[f[:,0]],p[f[:,2]]-p[f[:,0]]);valid=np.linalg.norm(dv,axis=1)>1e-11;f=f[valid];dv=dv[valid]
        # average the seam normals and the pole normals explicitly
        _,w=np.unique(np.round(p,5),axis=0,return_inverse=True)
        normals=np.zeros((w.max()+1,3),float)
        for j in range(3):np.add.at(normals,w[f[:,j]],dv)
        norm=np.linalg.norm(normals,axis=1,keepdims=True)
        normals[norm[:,0]<1e-15]=[0,1,0];normals/=np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-15)
        tu=m['tile']; im=tiles[tu]; uv=(m['uv']*np.array([im.width-1,im.height-1])+np.array(pos[tu])+.5)/np.array([width,height])
        mesh={'name':m['name'],'primitives':[{'attributes':{'POSITION':accessor(p,'VEC3',5126,34962,True),'NORMAL':accessor(normals[w].astype('<f4'),'VEC3',5126,34962),'TEXCOORD_0':accessor(uv.astype('<f4'),'VEC2',5126,34962)},'indices':accessor(f.ravel().astype('<u2'),'SCALAR',5123,34963),'material':md[mat]}]}
        nodes[groups[grp]]['children'].append(len(nodes));nodes.append({'name':m['name'],'mesh':len(meshes)});meshes.append(mesh)
    png=io.BytesIO();atlas.save(png,'PNG',compress_level=7)
    vi=buf(png.getvalue())
    while len(binbuf)%4:binbuf.append(0)
    j=dict(asset={'version':'2.0','generator':'Reference modeled character studio'},scene=0,scenes=[{'name':'Blue whale maid','nodes':roots}],nodes=nodes,meshes=meshes,materials=materials,textures=[{'sampler':0,'source':0}],samplers=[{'magFilter':9729,'minFilter':9987,'wrapS':33071,'wrapT':33071}],images=[{'name':'Painted costume and features','bufferView':vi,'mimeType':'image/png'}],accessors=acc,bufferViews=views,buffers=[{'byteLength':len(binbuf)}])
    data=json.dumps(j,ensure_ascii=False,separators=(',',':')).encode();data+=b' '*((-len(data))%4)
    binary=struct.pack('<4sII',b'glTF',2,28+len(data)+len(binbuf))+struct.pack('<I4s',len(data),b'JSON')+data+struct.pack('<I4s',len(binbuf),b'BIN\0')+binbuf
    (ROOT/'蓝鲸女仆.glb').write_bytes(binary)
    print(len(meshes),'meshes',sum(a['count']//3 for a in acc if a['type']=='SCALAR'),'triangles',atlas.size,len(binary),flush=True)
if __name__=='__main__':export()



