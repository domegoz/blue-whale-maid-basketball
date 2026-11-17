"""Retarget the reference take onto the existing whale maid armature."""
import json
from pathlib import Path
import struct
import sys

import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from rig_animation import ROOT, align, load_glb, rot, skeleton, smooth

R = skeleton()
BASE = ROOT / '骨骼动画/篮球尺寸校验/大篮球_直径1.75倍.glb'
OUT = ROOT / '骨骼动画/篮球舞_原片复刻'
GLB = OUT / '蓝鲸女仆_自我介绍与篮球舞.glb'
CLIP = 'Basketball Dance'
RADIUS = .124 * 1.75
IMPACTS = (12.90,13.767,14.467,15.4,15.9,16.567,17.433,18.267,19.233,21.4,22.367)
IDENTITY = Rotation.identity()


def norm(x):
    return x / max(1e-7, np.linalg.norm(x))


def wrap(x):
    return (x + 180) % 360 - 180


def filter_yaw(positions, a, b):
    direction = positions[:, a] - positions[:, b]
    angles = np.unwrap(np.arctan2(-direction[:, 2], direction[:, 0]))
    return np.rad2deg(gaussian_filter1d(median_filter(angles, 3), 1.35))


def fill_mask(data, mask):
    flat = data.reshape(len(data), -1).copy()
    for column in flat.T:
        valid = ~mask & np.isfinite(column)
        column[:] = np.interp(np.arange(len(data)), np.flatnonzero(valid), column[valid])
    return flat.reshape(data.shape)


def parabola(t, start, end, height):
    phase = np.clip((t - start) / (end - start), 0, 1)
    return height * 4 * phase * (1 - phase)


def rest_delta(name):
    parent = R.parent[R.index[name]]
    return R.point(name) - (R.point(R.names[parent]) if parent >= 0 else 0)


def forward(q, tr):
    positions, rotations = {}, {}
    for i, name in enumerate(R.names):
        p = R.parent[i]
        delta = tr.get(name, rest_delta(name))
        local = q.get(name, IDENTITY)
        if p < 0:
            positions[name], rotations[name] = delta, local
        else:
            parent = R.names[p]
            positions[name] = positions[parent] + rotations[parent].apply(delta)
            rotations[name] = rotations[parent] * local
    return positions, rotations


def follow_arm(rest, target, previous):
    # Transport the sleeve's roll along its path. A new shortest arc from the
    # bind pose at each frame has an undefined roll when the limb points up.
    if previous is None:
        return align(rest,target)
    carried = align(previous.apply(norm(rest)),target)*previous
    cosine = float(norm(rest)@target)
    if cosine < -.88:
        return carried
    toward_rest = (align(rest,target)*carried.inv()).as_rotvec() @ target
    roll = np.clip(.16*smooth(-.88,.2,cosine)*toward_rest,-.065,.065)
    return Rotation.from_rotvec(target*roll)*carried


def body_tracks(data, times):
    original = data['world'][:, :, :3] * [1, -1, -1]
    screen = data['screen']
    # In these front-facing lyric gestures the detector briefly reversed left/right.
    bad = ((times >= 42.67) & (times <= 42.96)) | ((times >= 43.58) & (times <= 43.76))
    bad |= (times >= 25.59) & (times <= 25.69)
    cleaned = median_filter(fill_mask(original, bad),size=(3,1,1),mode='nearest')
    point = gaussian_filter1d(cleaned,1.25,axis=0,mode='nearest')
    projected = gaussian_filter1d(fill_mask(screen, bad),1.15,axis=0,mode='nearest')
    shoulder_yaw = filter_yaw(point, 11, 12)
    hip_yaw = filter_yaw(point, 23, 24)
    face_yaw = filter_yaw(point, 7, 8)
    hip_screen = projected[:, [23, 24], :2].mean(axis=1)
    torso_size = np.linalg.norm((hip_screen - projected[:, [11, 12], :2].mean(axis=1))
                                * [16/9, 1], axis=1)
    scale = np.clip(.95 / gaussian_filter1d(torso_size, 7), 1.8, 4.7)
    root_x = gaussian_filter1d((hip_screen[:, 0] - .50) * scale, 1.4)
    jump = parabola(times, 28.68, 29.17, .39) + parabola(times, 33.60, 34.12, .53)
    jump += parabola(times, 34.37, 34.63, .07)
    chest_slope = []
    for i, p in enumerate(point):
        up = norm(p[[11, 12]].mean(0) - p[[23, 24]].mean(0))
        local = rot(y=shoulder_yaw[i]).inv().apply(up)
        chest_slope.append([np.clip(np.degrees(np.arctan2(local[2], local[1])), -38, 52),
                            np.clip(-np.degrees(np.arctan2(local[0], local[1])), -39, 39)])
    slopes = gaussian_filter1d(chest_slope, 1, axis=0)
    yaw_speed = gaussian_filter1d(np.gradient(shoulder_yaw), 1.4)
    palms, hip_positions, frames = [], [], []
    previous_arm = {}
    for i, (t, p) in enumerate(zip(times, point)):
        pitch, lean = slopes[i]
        # Preserve fast shoulder accents and let heavy hair follow several frames later.
        twist = np.clip(wrap(shoulder_yaw[i] - hip_yaw[i]), -50, 50)
        pelvis = rot(y=hip_yaw[i]) * rot(x=.19*pitch, z=.22*lean)
        q = {'Hips': pelvis, 'Spine': rot(y=.48*twist) * rot(x=.36*pitch,z=.35*lean),
             'Chest': rot(y=.52*twist) * rot(x=.43*pitch,z=.39*lean)}
        torso = pelvis * q['Spine'] * q['Chest']
        face = p[0] - p[[7, 8]].mean(0)
        nod = np.clip(-np.degrees(np.arctan2(face[1], np.linalg.norm(face[[0, 2]]))) - 15, -31, 31)
        head_yaw = shoulder_yaw[i] + np.clip(wrap(face_yaw[i] - shoulder_yaw[i]), -54, 54)
        ear = rot(y=head_yaw).inv().apply(p[7] - p[8])
        tilt = np.clip(np.degrees(np.arctan2(ear[1], ear[0])), -30, 30)
        desired_head = rot(y=head_yaw) * rot(x=.63*nod+.19*pitch, z=.68*tilt+.13*lean)
        head_local = torso.inv() * desired_head
        q['Neck'] = Rotation.from_rotvec(.27 * head_local.as_rotvec())
        q['Head'] = q['Neck'].inv() * head_local
        lag = max(i-3, 0)
        sway = np.clip(yaw_speed[lag], -15, 15)
        past_pitch, past_lean = slopes[lag]
        lower = np.clip(pitch/64, 0, 1)
        tr = {'Hips': np.array([root_x[i], .70, .12]),
              'SkirtHem': rest_delta('SkirtHem') + [0, .046*lower, 0]}
        q.update({'Skirt':rot(x=-.11*past_pitch,y=-.43*sway,z=-.16*past_lean),
                  'SkirtHem':rot(x=-.08*past_pitch,y=-.38*sway,z=-.16*past_lean),
                  'Cowlick':rot(x=.22*(past_pitch-pitch),z=.3*(past_lean-lean),y=-.45*sway),
                  'TailBase':rot(x=.16*past_pitch,y=-.6*sway,z=.13*past_lean),
                  'TailMid':rot(x=.13*past_pitch,y=-.5*sway,z=.13*past_lean),
                  'TailFlukes':rot(x=.12*past_pitch,y=-.35*sway,z=.09*past_lean)})
        for part in ('R','C','L'):
            q['HairUpper.'+part] = rot(x=-.12*past_pitch,y=-.47*sway,z=-.16*past_lean)
            q['HairTip.'+part] = rot(x=-.10*past_pitch,y=-.53*sway,z=-.23*past_lean)
        for side, k, sign in [('L',0,1), ('R',1,-1)]:
            suff = '.'+side
            a, e, h = 11+k, 13+k, 15+k
            u = torso.inv().apply(norm(p[e]-p[a]))
            f = torso.inv().apply(norm(p[h]-p[e]))
            up = smooth(-.19,.24,p[h,1]-p[a,1])
            across = smooth(-.08,.18, -sign*(p[h,0]-p[a,0]))
            # The maid's chin occupies the adult performer's raised-hand space.
            # Bring cheek gestures forward and open the raised elbow slightly.
            u = norm(u + [sign*(.09+.31*up), -.06*up, .20+.38*up])
            f = norm(f + [sign*.08*up, -.05*up, .13+.38*up+.12*across])
            shr = np.clip((p[a,1]-p[12-k,1])*.29,-.065,.065)
            reach = max(up, smooth(.50,.93,abs(u[0])))
            tr['Clavicle'+suff] = rest_delta('Clavicle'+suff)+[sign*(.019+.062*reach),shr,.06+.145*up]
            tr['Forearm'+suff] = rest_delta('Forearm'+suff)*(1+.34*reach)
            arm,fore = 'UpperArm'+suff,'Forearm'+suff
            upper = follow_arm(rest_delta(fore),u,previous_arm.get(arm))
            lower_arm = follow_arm(rest_delta('Hand'+suff),f,previous_arm.get(fore))
            previous_arm[arm],previous_arm[fore] = upper,lower_arm
            q['UpperArm'+suff] = upper
            q['Forearm'+suff] = upper.inv()*lower_arm
            q['Hand'+suff] = rot(x=-7*up, z=sign*(3+6*up))
            upper_leg = norm(p[25+k]-p[23+k])
            shin = norm(p[27+k]-p[25+k])
            conf = projected[i,27+k,3]
            plausible = 1 - smooth(1.02,1.24,projected[i,27+k,1])
            full_body = smooth(12.80,13.20,t)*(1-smooth(46.5,47.1,t))
            observed = float(smooth(.09,.65,conf)*plausible*full_body)
            kick = max(smooth(40.43,40.64,t)*(1-smooth(41.20,41.48,t)),
                       smooth(41.69,41.92,t)*(1-smooth(42.13,42.34,t)))
            observed = max(observed, .82*kick)
            ready = norm(pelvis.apply([sign*.145,-.97,.19+.15*lower]))
            planted = norm(pelvis.apply([sign*.015,-.99,-.14-.16*lower]))
            upper_leg = norm(ready*(1-observed)+upper_leg*observed)
            shin = norm(planted*(1-observed)+shin*observed)
            # Work in pelvis space to keep the back of the stockings facing backward on turns.
            leg = align(rest_delta('Shin'+suff),pelvis.inv().apply(upper_leg))
            calf = align(rest_delta('Foot'+suff),pelvis.inv().apply(shin))
            q['Thigh'+suff] = leg
            q['Shin'+suff] = leg.inv()*calf
            toes = p[31+k]-p[29+k]
            foot_yaw = hip_yaw[i] + np.clip(wrap(np.degrees(np.arctan2(toes[0],toes[2]))-hip_yaw[i]),-34,34)*observed
            q['Foot'+suff] = (pelvis*calf).inv()*rot(y=foot_yaw)
        if t<9.05:
            from opening_acting import opening_pose
            opening_pose(t,q,tr,rest_delta,head_yaw,tilt,nod,lean)
        pos, ori = forward(q,tr)
        low = min(pos['Foot.R'][1],pos['Foot.L'][1])
        tr['Hips'][1] += .104-low+jump[i]
        compression = np.clip(.645-tr['Hips'][1],0,.24)
        tr['Skirt'] = rest_delta('Skirt')+[0,.24*compression,0]
        tr['SkirtHem'] += [0,.45*compression,0]
        for name in ('Hand.L','Hand.R'):
            pos[name][1] += .104-low+jump[i]
        palms.append([pos[name]+ori[name].apply([side*.012,-.035,.016])
                      for name,side in [('Hand.L',1),('Hand.R',-1)]])
        hip_positions.append(tr['Hips'].copy())
        frames.append((q,tr))
    return frames, np.array(palms), np.array(hip_positions), projected, hip_screen


def ball_tracks(data, times, palms, hips, projected, screen_hip):
    measured = data['ball']
    valid = np.isfinite(measured[:,0]) & (times>=10.666) & (times<=23.80)
    screen = fill_mask(measured[:,:3], ~valid)
    pixscale = .085 / np.clip(gaussian_filter1d(median_filter(screen[:,2], 7), 3), .023, .09)
    base = hips.copy()
    base[:,0] += (screen[:,0]-screen_hip[:,0])*pixscale
    base[:,1] += (screen_hip[:,1]-screen[:,1])*pixscale*9/16
    base[:,2] += .37
    anchors = np.zeros((len(times),3))
    known = np.zeros(len(times),bool)
    for i in np.flatnonzero(valid):
        offsets = (screen[i,:2]-projected[i,[15,16],:2])*[1,-9/16]
        distances = np.linalg.norm(offsets,axis=1)/screen[i,2]
        if distances.min() > 1.85:
            continue
        candidates = []
        for k, offset in enumerate(offsets):
            direction = norm(np.r_[offset / max(screen[i,2], .005), .80])
            candidates.append(palms[i,k]+direction*(RADIUS+.016))
        weights = np.exp(-3.4*(distances-distances.min()))
        target = np.average(candidates,axis=0,weights=weights)
        anchors[i] = target-base[i]
        known[i] = True
    # Each impact is a measured reversal of the source ball at the floor.
    for t in IMPACTS:
        i = np.argmin(abs(times-t))
        known[max(0,i-2):i+3] = False
    residual = fill_mask(anchors,~known)
    for t in IMPACTS:
        i = np.argmin(abs(times-t))
        anchors[i] = residual[i]
        anchors[i,1] = RADIUS - base[i,1]
        known[i] = True
    pos = base + fill_mask(anchors,~known)
    pos[:,1] = np.maximum(pos[:,1],RADIUS)
    # The enlarged ball travels around the front edge of the maid's bell skirt on low crossovers.
    front = smooth(12.60,12.95,times)*(1-smooth(19.34,19.58,times))
    front += smooth(22.08,22.30,times)*(1-smooth(22.44,22.62,times))
    low = 1-smooth(.35,.63,pos[:,1])
    half_depth = .60*np.sqrt(np.maximum(0,1-((pos[:,0]-hips[:,0])/.71)**2))
    pos[:,2] += front*low*np.maximum(0,hips[:,2]+half_depth-pos[:,2])
    # Enter and leave the full-body camera along the same screen edges as the reference.
    for start,end,outside,entry in [(10.39,10.70,[-2.9,2.52,.49],True),
                                     (23.67,24.04,[-3.05,2.46,.53],False)]:
        edge = np.argmin(abs(times - (end if entry else start)))
        delta = (np.array(outside)-pos[edge]).copy()
        phase = np.clip((times-start)/(end-start),0,1)
        if entry:
            phase = 1-phase
        region = times < end if entry else times > start
        pos[region] = pos[edge] + phase[region,None]*delta
        pos[region,1] += .14*np.sin(np.pi*phase[region])
    visible = ((times>=10.39)&(times<=24.04)).astype(float)
    return pos, visible


def export(frames, times, ball_pos, visible):
    from secondary_motion import add_details, add_morphs
    detail_controls=add_details(frames,times,ball_pos)
    from garment_dynamics import simulate
    doc, buf = load_glb(BASE)
    from rig_animation import read_accessor
    cloth_meshes=[read_accessor(doc,buf,m['primitives'][0]['attributes']['POSITION']) for m in doc['meshes']
                 if m['name'].startswith(('Midnight embroidered','Scalloped ivory','Skirt underside','Skirt ribbon','Whale apron'))]
    detail_controls.update(simulate(frames,times,ball_pos,visible,R,cloth_meshes))
    from front_hair import simulate_front
    detail_controls['front_hair']=simulate_front(doc,buf,detail_controls.pop('head_pose'),times,R)
    for q,_ in frames:
        q['Cowlick']=IDENTITY
    old_animation = doc['animations'][0]
    first = min(s['input'] for s in old_animation['samplers'])
    unused_view = doc['accessors'][first]['bufferView']
    # This authored base packs its old clip after all meshes. Preserve the full rig and atlas.
    discard = set(range(first,len(doc['accessors'])))
    old_refs = {s[k] for s in old_animation['samplers'] for k in ('input','output')}
    assert discard == old_refs
    cut = doc['bufferViews'][unused_view]['byteOffset']
    assert all(v['byteOffset'] >= cut for v in doc['bufferViews'][unused_view:])
    doc['accessors'] = doc['accessors'][:first]
    doc['bufferViews'] = doc['bufferViews'][:unused_view]
    del buf[cut:]
    def acc(data,kind,bounds=False,component=5126):
        ar = np.ascontiguousarray(data,dtype={5126:'<f4',5123:'<u2',5121:'u1'}[component])
        buf.extend(bytes(-len(buf)%4))
        view = len(doc['bufferViews'])
        doc['bufferViews'].append(dict(buffer=0,byteOffset=len(buf),byteLength=ar.nbytes))
        buf.extend(ar.tobytes())
        a = dict(bufferView=view,componentType=component,type=kind,count=len(ar))
        if bounds: a.update(min=ar.reshape(len(ar),-1).min(0).tolist(),max=ar.reshape(len(ar),-1).max(0).tolist())
        doc['accessors'].append(a)
        return len(doc['accessors'])-1
    timeacc = acc(times-times[0],'SCALAR',True)
    nodes = {n['name']:i for i,n in enumerate(doc['nodes'])}
    tracks = {}
    spin = np.cumsum(np.r_[0,np.linalg.norm(np.diff(ball_pos,axis=0),axis=1)]) / RADIUS
    for i,(q,tr) in enumerate(frames):
        q['Basketball'] = rot(x=np.degrees(spin[i])*.70,y=-24+np.degrees(spin[i])*.15,z=18)
        tr['Basketball'] = ball_pos[i]
        for name,v in q.items(): tracks.setdefault((name,'rotation'),[]).append(v.as_quat())
        for name,v in tr.items(): tracks.setdefault((name,'translation'),[]).append(v)
    tracks['Basketball','scale'] = np.repeat(visible[:,None],3,axis=1)
    channels, samplers = [], []
    for (name,path),values in tracks.items():
        values = np.asarray(values)
        if path == 'rotation':
            for i in range(1,len(values)):
                if values[i-1] @ values[i] < 0: values[i] *= -1
        idx = acc(values,'VEC4' if path=='rotation' else 'VEC3')
        channels.append(dict(sampler=len(samplers),target=dict(node=nodes[name],path=path)))
        samplers.append(dict(input=timeacc,output=idx,interpolation='LINEAR'))
    add_morphs(doc,buf,acc,timeacc,nodes,channels,samplers,detail_controls)
    doc['animations'] = [dict(name=CLIP,channels=channels,samplers=samplers)]
    buf.extend(bytes(-len(buf)%4))
    doc['buffers'][0]['byteLength'] = len(buf)
    meta = json.dumps(doc,ensure_ascii=False,separators=(',',':')).encode()
    meta += b' '*(-len(meta)%4)
    out = struct.pack('<4sII',b'glTF',2,28+len(meta)+len(buf))+struct.pack('<I4s',len(meta),b'JSON')+meta
    OUT.mkdir(exist_ok=True)
    tmp = GLB.with_suffix('.tmp')
    tmp.write_bytes(out+struct.pack('<I4s',len(buf),b'BIN\0')+buf)
    tmp.replace(GLB)


if __name__ == '__main__':
    from motion_curves import finish_body, finish_ball
    dance=np.load(HERE/'tracking.npz')
    opening=np.load(HERE/'opening_performance.npz')
    intro_count=round(float(dance['start'])*float(dance['fps']))
    data={key:np.concatenate((opening[key][:intro_count],dance[key])) for key in ('world','screen')}
    data['ball']=np.concatenate((np.full((intro_count,dance['ball'].shape[1]),np.nan),dance['ball']))
    times=np.arange(len(data['world']))/float(dance['fps'])
    frames,_,_,projected,hip_screen = body_tracks(data,times)
    dense=np.linspace(times[0],times[-1],2*(len(times)-1)+1)
    frames,palms,hips = finish_body(frames,times,dense,R)
    pos,_ = ball_tracks(data,times,palms[::2],hips[::2],projected,hip_screen)
    ball,visible = finish_ball(pos,times,dense,IMPACTS,RADIUS)
    export(frames,dense,ball,visible)
    print(GLB, len(frames), 'poses')
