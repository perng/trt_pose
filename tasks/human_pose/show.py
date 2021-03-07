import time, sys
t0 = time.time()
from torch2trt import TRTModule
import torch
import cv2
import torchvision.transforms as transforms
import PIL.Image
from trt_pose.draw_objects import DrawObjects
from trt_pose.parse_objects import ParseObjects
import json
import trt_pose.coco

t1 = time.time()
print('import:', t1 - t0)
t0 = t1

OPTIMIZED_MODEL = 'resnet18_baseline_att_224x224_A_epoch_249_trt.pth'

model_trt = TRTModule()
model_trt.load_state_dict(torch.load(OPTIMIZED_MODEL))

t1 = time.time()
print('load model:', t1 - t0)
t0 = t1

WIDTH = 224
HEIGHT = 224

data = torch.zeros((1, 3, HEIGHT, WIDTH)).cuda()

torch.cuda.current_stream().synchronize()
#for i in range(50):
#    y = model_trt(data)
#torch.cuda.current_stream().synchronize()

#t1 = time.time()
#print('50 trt round:', t1-t0)
#t0 = t1

mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
device = torch.device('cuda')

t1 = time.time()
print('set device:', t1-t0)
t0 = t1

def preprocess(image):
    global device
    device = torch.device('cuda')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = PIL.Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(device)
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]


with open('human_pose.json', 'r') as f:
    human_pose = json.load(f)

    print(human_pose)

topology = trt_pose.coco.coco_category_to_topology(human_pose)
print('topology', topology)
parse_objects = ParseObjects(topology)
print('parse_objects', parse_objects)
draw_objects = DrawObjects(topology)
print('draw_objects', draw_objects)

from jetcam.usb_camera import USBCamera
# from jetcam.csi_camera import CSICamera
from jetcam.utils import bgr8_to_jpeg

camera = USBCamera(capture_device=1, width=WIDTH, height=HEIGHT, capture_fps=30)
# camera = CSICamera(width=WIDTH, height=HEIGHT, capture_fps=30)

camera.running = True

t1 = time.time()
print('camera running:', t1-t0)
t0 = t1


#import ipywidgets
#from IPython.display import display

#image_w = ipywidgets.Image(format='jpeg')

#display(image_w)

frame = 0
t0 = time.time()
def execute(change):
    global frame, t0, t1
    image = change['new']
    data = preprocess(image)
    cmap, paf = model_trt(data)
    cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
    counts, objects, peaks = parse_objects(cmap, paf)#, cmap_threshold=0.15, link_threshold=0.15)
    draw_objects(image, counts, objects, peaks)
    frame += 1
    if frame % 10 == 0:
        #print(cmap, paf, counts, objects, peaks)
        #print('cmap', cmap)
        #print('paf', paf)
        print('counts', counts[0])
        print('objects', objects.shape)
        print('peaks', peaks.shape)
        
        t1 = time.time()
        print('FPS = ', frame/(t1-t0))
    cv2.imshow('monitor', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        sys.exit(0)

execute({'new': camera.value})

camera.observe(execute, names='value')

#camera.unobserve_all()



