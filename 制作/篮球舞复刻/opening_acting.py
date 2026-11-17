"""Perform the close-up in the original timebase, including the untracked bow."""
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation

from rig_animation import align, rot, smooth


def opening_pose(t,q,tr,rest,head_yaw,tilt,nod,lean):
    intro=1-smooth(8.42,9.05,t)
    bow=1-smooth(.33,.96,t)
    standing=1-bow
    head_yaw=((head_yaw+180)%360-180)*smooth(.67,.96,t)
    body_yaw=standing*np.clip(head_yaw*.23,-6,6)
    def blend(old,new,weight):
        return old*Rotation.from_rotvec((old.inv()*new).as_rotvec()*weight)
    def aim(name,pose,weight=intro):
        q[name]=blend(q[name],pose,weight)
    pelvis=rot(x=14*bow,y=body_yaw)
    spine=rot(x=20*bow+.9*standing,z=standing*.06*lean)
    chest=rot(x=18*bow,y=body_yaw*.6)
    neck=rot(x=6*bow+standing*np.clip(nod*.12,-2,2))
    head=rot(x=15*bow+standing*np.clip(nod*.52,-10,12),
             y=standing*np.clip(head_yaw*.52,-17,17),z=standing*np.clip(tilt*.6,-8,8))
    for name,pose in zip(('Hips','Spine','Chest','Neck','Head'),(pelvis,spine,chest,neck,head)):
        aim(name,pose)
    tr['Hips'][0]*=1-intro*.86
    torso=pelvis*spine*chest
    for side,sign in [('L',1),('R',-1)]:
        suffix='.'+side
        arm_hold=1-smooth(7.34,8.05,t)
        upper=align(rest('Forearm'+suffix),[sign*.34,-.91,.18+.11*bow])
        fore=align(rest('Hand'+suffix),[sign*.08,-.95,.32])
        aim('UpperArm'+suffix,torso.inv()*upper,arm_hold)
        aim('Forearm'+suffix,upper.inv()*fore,arm_hold)
        aim('Hand'+suffix,rot(z=sign*7),arm_hold)
        for name,delta in [('Clavicle',[sign*.024,0,.056]),('Forearm',[0,0,0])]:
            key=name+suffix
            tr[key]+=arm_hold*(rest(key)+delta-tr[key])
        if side=='R':
            cue=smooth(7.76,8.09,t)*(1-smooth(8.22,8.52,t))
            upper=align(rest('Forearm.R'),[-.61,-.30,.73])
            fore=align(rest('Hand.R'),[-.23,.88,.43])
            aim('UpperArm.R',torso.inv()*upper,cue)
            aim('Forearm.R',upper.inv()*fore,cue)
            tr['Clavicle.R']+=cue*np.array([-.051,.040,.055])
        thigh=align(rest('Shin'+suffix),[sign*.13,-.982,.14+.11*bow])
        shin=align(rest('Foot'+suffix),[sign*.023,-.982,-.16-.08*bow])
        aim('Thigh'+suffix,pelvis.inv()*thigh)
        aim('Shin'+suffix,thigh.inv()*shin)
        aim('Foot'+suffix,shin.inv())


def facial_performance(t,dance_eyes,dance_mouth):
    face=np.load(Path(__file__).with_name('opening_performance.npz'))['face']*[1,9/16,1]
    def dist(a,b):
        return np.linalg.norm(face[:,a,:2]-face[:,b,:2],axis=1)
    def sample(x,sigma=.59):
        good=np.isfinite(x)
        grid=np.arange(len(x))/30
        fitted=np.interp(grid,grid[good],x[good])
        return np.interp(t,grid,gaussian_filter1d(fitted,sigma,mode='nearest'))
    spacing=dist(33,263)
    opening=np.clip((sample(dist(13,14)/spacing)-.010)/.254,0,1)
    width=sample(dist(61,291)/spacing)
    lip_wide=smooth(.474,.615,width)
    narrow=1-smooth(.405,.547,width)
    speaking=smooth(.47,.78,t)
    jaw=opening**.84*speaking
    a=jaw*(1-.57*narrow)*(1-.36*lip_wide)
    e=lip_wide*(.20+.77*jaw)*(1-.56*a)*speaking
    u=narrow*jaw*(1-.30*a)
    closed=(1-smooth(.018,.12,opening))*.80*speaking
    speech=np.stack((a,e,u,closed),axis=1)
    speech/=np.maximum(1,speech.sum(1,keepdims=True))
    timeline=[0,.45,.98,1.36,1.91,2.63,3.38,4.03,4.91,5.54,6.22,6.88,7.50,8.08,8.54,9.08]
    smile=PchipInterpolator(timeline,[.10,.13,.55,.80,.47,.25,.42,.23,.61,.41,.67,.44,.81,.36,.76,.47])(np.clip(t,0,9.08))
    brow=sample((dist(105,159)/dist(33,133)+dist(334,386)/dist(362,263))/2,1.25)
    brow=np.clip((brow-.80)*3.4,-.55,.80)+.32*smile
    eyesize=sample((dist(159,145)/dist(33,133)+dist(386,374)/dist(362,263))/2,.46)
    blink=(1-smooth(.055,.235,eyesize))*speaking
    eye_open=smooth(.29,.398,eyesize)
    def eye_fraction(iris,inner,outer,axis):
        return (face[:,iris,axis]-(face[:,inner,axis]+face[:,outer,axis])*.5)/dist(inner,outer)
    gx=sample((eye_fraction(468,33,133,0)+eye_fraction(473,263,362,0))/2,1.0)
    gy=sample((eye_fraction(468,33,133,1)+eye_fraction(473,263,362,1))/2,1.0)
    gx=np.clip((gx+.012)*6,-.75,.75)
    gy=np.clip((gy+.055)*7,-.70,.75)
    gaze_down=(1-smooth(.54,1.12,t))*.62
    eyes=np.column_stack((blink,gx*(1-blink)*speaking,(gy*speaking+gaze_down)*(1-blink),
                          (1-smile)*.30*(1-blink),smile*(1-blink),.61*eye_open*(1-blink),
                          np.zeros(len(t)),np.clip(brow,-.8,1)*(1-.75*blink)))
    mouth=np.column_stack((.08*(1-jaw),.57*smile*(1-speech.sum(1)),np.zeros(len(t)),speech,
                           smile*(.27+.67*speaking)*(1-.74*u)))
    dance_eyes=np.column_stack((dance_eyes,np.clip(.62*dance_eyes[:,5]-.42*dance_eyes[:,3]+.31*dance_eyes[:,4],-.8,1)))
    dance_mouth=np.column_stack((dance_mouth,np.zeros_like(speech),dance_mouth[:,1]*.43))
    w=(1-smooth(8.51,9.06,t))[:,None]
    return eyes*w+dance_eyes*(1-w),mouth*w+dance_mouth*(1-w)
