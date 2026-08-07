import io
import math
import cv2
import numpy as np
from PIL import Image
from pillow_heif import register_heif_opener

register_heif_opener()

ROI_FRACTION = 0.60
NORMALIZED_BEAKER_SIZE = 600
NORMALIZED_BEAKER_RADIUS = 270
OUTLIER_SIGMA = 3.0
GAMMA = 2.2
EPSILON = 1e-6

FEATURE_ORDER = [
    "Clarity_Raw","ROI_Mean_Raw","ROI_SD_Raw","Center_Clarity_Raw",
    "Mid_Clarity_Raw","Outer_Clarity_Raw","Center_Mid_Contrast_Raw",
    "Center_Outer_Contrast_Raw","RCDI_Raw","COR_Raw","Center_Clarity_Z",
    "Center_Mid_Contrast_Z","Center_Outer_Contrast_Z","Center_GEC","LuED",
    "Radial_Gradient","RSI_Correlation","RSI_RadialCV",
]

def decode_image(image_bytes, filename=""):
    suffix = filename.lower().rsplit(".",1)[-1] if "." in filename else ""
    if suffix in {"heic","heif"}:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return img

def circle_edge_support(edge_image,cx,cy,radius,band_width=4,n_angles=360):
    h,w=edge_image.shape
    hits=0
    for theta in np.linspace(0,2*np.pi,n_angles,endpoint=False):
        found=False
        for offset in range(-band_width,band_width+1):
            r=radius+offset
            x=int(round(cx+r*np.cos(theta)))
            y=int(round(cy+r*np.sin(theta)))
            if 0<=x<w and 0<=y<h and edge_image[y,x]>0:
                found=True; break
        hits += int(found)
    return hits/n_angles if n_angles else 0.0

def circle_inside_fraction(cx,cy,radius,width,height):
    left,right=cx-radius,cx+radius
    top,bottom=cy-radius,cy+radius
    inside_x=max(0,min(right,width)-max(left,0))
    inside_y=max(0,min(bottom,height)-max(top,0))
    box_area=(2*radius)**2
    return min(1.0,(inside_x*inside_y)/box_area) if box_area>0 else 0.0

def detect_beaker_circle(img):
    h0,w0=img.shape[:2]
    scale=min(1.0,1000.0/max(h0,w0))
    detect=cv2.resize(img,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA) if scale<1 else img.copy()
    h,w=detect.shape[:2]
    gray=cv2.cvtColor(detect,cv2.COLOR_BGR2GRAY)
    blurred=cv2.GaussianBlur(gray,(9,9),2)
    edges=cv2.Canny(blurred,50,140)
    min_r=int(min(h,w)*0.12); max_r=int(min(h,w)*0.48)
    circles=cv2.HoughCircles(
        blurred,cv2.HOUGH_GRADIENT,dp=1.2,minDist=min(h,w)*0.15,
        param1=120,param2=28,minRadius=min_r,maxRadius=max_r
    )
    if circles is not None:
        scored=[]
        icx,icy=w/2,h/2
        for x,y,r in np.round(circles[0]).astype(int):
            es=circle_edge_support(edges,x,y,r,4)
            cd=math.sqrt((x-icx)**2+(y-icy)**2)/max(h,w)
            cs=max(0.0,1.0-cd/0.45)
            rf=r/min(h,w)
            rs=1.0 if 0.18<=rf<=0.42 else (0.60 if 0.14<=rf<=0.46 else 0.20)
            ins=circle_inside_fraction(x,y,r,w,h)
            score=0.65*es+0.15*cs+0.10*rs+0.10*ins
            scored.append((score,x,y,r,es,ins))
        scored.sort(reverse=True)
        _,x,y,r,es,ins=scored[0]
        flag="OK" if es>=0.45 and ins>=0.90 else ("REVIEW" if es>=0.30 else "LOW_CONFIDENCE")
        inv=1.0/scale
        return x*inv,y*inv,r*inv,"HoughV2",flag

    contours,_=cv2.findContours(edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    center=np.array([w/2,h/2]); candidates=[]
    for contour in contours:
        area=cv2.contourArea(contour)
        if area<=0: continue
        (x,y),r=cv2.minEnclosingCircle(contour)
        if r<min_r or r>max_r: continue
        fill=area/(math.pi*r*r+EPSILON)
        dist=np.linalg.norm(np.array([x,y])-center)
        candidates.append((dist/max(h,w)+abs(0.6-fill),x,y,r))
    if candidates:
        candidates.sort(key=lambda z:z[0])
        _,x,y,r=candidates[0]; inv=1.0/scale
        return x*inv,y*inv,r*inv,"Contour","REVIEW"

    return w0/2,h0/2,min(h0,w0)*0.25,"CenterFallback","REVIEW"

def standardize_beaker(img,cx,cy,radius):
    s=NORMALIZED_BEAKER_RADIUS/radius
    M=np.array([
        [s,0,NORMALIZED_BEAKER_SIZE/2-s*cx],
        [0,s,NORMALIZED_BEAKER_SIZE/2-s*cy]
    ],dtype=np.float32)
    return cv2.warpAffine(
        img,M,(NORMALIZED_BEAKER_SIZE,NORMALIZED_BEAKER_SIZE),
        flags=cv2.INTER_AREA,borderMode=cv2.BORDER_CONSTANT,borderValue=0
    )

def make_roi_mask(shape):
    h,w=shape; cx,cy=w//2,h//2
    r=int(NORMALIZED_BEAKER_RADIUS*ROI_FRACTION)
    Y,X=np.ogrid[:h,:w]
    return (X-cx)**2+(Y-cy)**2<=r**2

def radial_coordinates(shape,mask):
    h,w=shape; cx,cy=w//2,h//2
    Y,X=np.ogrid[:h,:w]
    r=np.sqrt((X-cx)**2+(Y-cy)**2)
    return r/(np.max(r[mask])+EPSILON)

def safe_mean(v):
    return float(np.mean(v)) if len(v) else float("nan")

def compute_gec(gray,mask):
    gx=cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=3)
    gy=cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=3)
    g=np.sqrt(gx**2+gy**2)
    return float(np.mean(g[mask])) if np.sum(mask) else float("nan")

def compute_lued(gray,mask):
    gx=cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=3)
    gy=cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=3)
    g=np.sqrt(gx**2+gy**2)
    return float(np.var(g[mask])) if np.sum(mask) else float("nan")

def compute_radial_gradient(gray,mask,rn):
    if np.sum(mask)<10: return float("nan")
    return float(np.polyfit(rn[mask],gray[mask],1)[0])

def compute_rsi(gray,mask,rn):
    r=rn[mask]; i=gray[mask]
    if len(r)<10: return float("nan")
    rs=np.std(r); isd=np.std(i)
    if rs<EPSILON or isd<EPSILON: return float("nan")
    rz=(r-np.mean(r))/(rs+EPSILON); iz=(i-np.mean(i))/(isd+EPSILON)
    return float(np.mean(rz*iz))

def calculate_metrics(standardized_img):
    gray=cv2.cvtColor(standardized_img,cv2.COLOR_BGR2GRAY).astype(np.float64)/255.0
    gray=np.power(gray,GAMMA)
    roi=make_roi_mask(gray.shape)
    vals=gray[roi]; mean=np.mean(vals); sd=np.std(vals)
    z=(gray-mean)/(sd+EPSILON)
    usable=roi&(np.abs(z)<OUTLIER_SIGMA)
    if np.sum(usable)<100: raise ValueError("Too few usable ROI pixels.")
    rn=radial_coordinates(gray.shape,usable)
    center=(rn<0.25)&usable
    mid=(rn>0.30)&(rn<0.45)&usable
    outer=(rn>0.60)&(rn<0.80)&usable
    cr,mr,orr=safe_mean(gray[center]),safe_mean(gray[mid]),safe_mean(gray[outer])
    cz,mz,oz=safe_mean(z[center]),safe_mean(z[mid]),safe_mean(z[outer])
    co=cr-orr; cm=cr-mr
    metrics={
        "Clarity_Raw":safe_mean(gray[usable]),
        "ROI_Mean_Raw":float(mean),"ROI_SD_Raw":float(sd),
        "Center_Clarity_Raw":cr,"Mid_Clarity_Raw":mr,"Outer_Clarity_Raw":orr,
        "Center_Mid_Contrast_Raw":cm,"Center_Outer_Contrast_Raw":co,
        "RCDI_Raw":co,"COR_Raw":cr/(orr+EPSILON),
        "Center_Clarity_Z":cz,"Center_Mid_Contrast_Z":cz-mz,
        "Center_Outer_Contrast_Z":cz-oz,
        "Center_GEC":compute_gec(z,center),
        "LuED":compute_lued(z,usable),
        "Radial_Gradient":compute_radial_gradient(z,usable,rn),
        "RSI_Correlation":compute_rsi(z,usable,rn),
        "RSI_RadialCV":np.std(gray[usable])/(abs(np.mean(gray[usable]))+EPSILON),
    }
    out={}
    for k in FEATURE_ORDER:
        v=float(metrics[k])
        out[k]=v if np.isfinite(v) else None
    return out

def preprocess_and_measure(image_bytes,filename=""):
    img=decode_image(image_bytes,filename)
    if img is None or img.size==0: raise ValueError("Image could not be decoded.")
    cx,cy,radius,method,flag=detect_beaker_circle(img)
    standardized=standardize_beaker(img,cx,cy,radius)
    metrics=calculate_metrics(standardized)
    qc={
        "detection_method":method,"detection_flag":flag,
        "beaker_center_x":float(cx),"beaker_center_y":float(cy),
        "beaker_radius_original":float(radius),"roi_fraction":ROI_FRACTION,
    }
    return metrics,qc
