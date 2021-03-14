import time, sys, logging, psutil
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
import pickle
import numpy as np
import sqlite3
from sqlite3 import Error

logging.basicConfig(filename='data.log', level=logging.DEBUG)

def adapt_array(arr):
    """
    http://stackoverflow.com/a/31312102/190597 (SoulNibbler)
    """
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())

def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)


# Converts np.array to TEXT when inserting
sqlite3.register_adapter(np.ndarray, adapt_array)

# Converts TEXT to np.array when selecting
sqlite3.register_converter("array", convert_array)

sql = ''' insert into data (timestamp, count, objects, peaks) values (?,?,?,?) '''
def create_connection(db_file):
    """ create a database connection to a SQLite database """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except Error as e:
        print(e)
    return conn


def process(data_queue):
    mem = psutil.virtual_memory()._asdict()
    logging.info("Before commit: percent: %f\t free:%d" % (mem['percent'],mem['free']))
    conn = create_connection('posedata.db')
    for d in data_queue:
        conn.cursor().execute(sql, d)
    conn.commit()
    logging.info("After commit: percent: %f\t free:%d" % (mem['percent'],mem['free']))
    conn.close()




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
parse_objects = ParseObjects(topology)
draw_objects = DrawObjects(topology)

from jetcam.usb_camera import USBCamera
from jetcam.utils import bgr8_to_jpeg

camera = USBCamera(capture_device=1, width=WIDTH, height=HEIGHT, capture_fps=30)
camera.running = True

t1 = time.time()
t0 = t1


frame = 0
t0 = time.time()
conn = None
data_queue = []
def execute(change):
    global frame, t0, t1, conn


    image = change['new']
    data = preprocess(image)
    cmap, paf = model_trt(data)
    cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
    # counts is the number of people
    counts, objects, peaks = parse_objects(cmap, paf)#, cmap_threshold=0.15, link_threshold=0.15)
    draw_objects(image, counts, objects, peaks)
    frame += 1
    if frame % 50 == 0:
        #print(cmap, paf, counts, objects, peaks)
        #print('cmap', cmap)
        #print('paf', paf)
        #print('counts', counts, counts.shape, counts[0])
        #print('objects', objects[0][0:3])
        #print('peaks', peaks[0][0][0:3])
        
        t1 = time.time()
        #print('FPS = ', frame/(t1-t0))
    count = counts.numpy()[0]
    if count > 0:
        data_queue.append((time.time(), count, objects[:,:count,:].numpy().tobytes(), peaks[:,:,:count,:].numpy().tobytes()))
        
    if len(data_queue) > 100:
        process(data_queue)
        #mem = psutil.virtual_memory()._asdict()
        #logging.info("Before commit: percent: %f\t free:%d" % (mem['percent'],mem['free']))
        data_queue.clear()

    cv2.imshow('monitor', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        sys.exit(0)

execute({'new': camera.value})

camera.observe(execute, names='value')

#camera.unobserve_all()



