path2scripts = 'd:\\projects\\bitbucket_lotgon\\NewWorld\\venv\\models\\research\\' # TODO: provide pass to the research folder
path2config ='.\\exported_models\\efficientdet_d0_coco17_tpu-32\\pipeline.config'
path2modelCheckpoint = "d:\\projects\\bitbucket_lotgon\\NewWorld\\venv\\workspace\\exported_models\\efficientdet_d0_coco17_tpu-32\\checkpoint\\"
path2label_map = '.\\data\\allImages\\Converted\\Duck-TFRecords-export\\tf_label_map.pbtxt ' # TODO: provide a path to the label map file
windowName = "New World"

import random, time, sys, os
import numpy as np
sys.path.append(".")
from capturer import Capturer
from pywinauto import keyboard
from mouse import WindMouse 
from ObjectDetection import ObjectDetection
sys.path.insert(0, path2scripts) # making scripts in models/research available for import



od = ObjectDetection(path2config, path2modelCheckpoint, path2label_map, cuda_device="-1")

captur = Capturer(".*%s.*" % windowName)
captur.ShowWindow()
WindMouse.wind_mouse(0, -600)
WindMouse.wind_mouse(0, 100)

while True:
    start = time.perf_counter()
    img = captur.Capture()
    print("Capture took %i seconds"%(time.perf_counter() - start))
    width, height = img.size 

    #start = time.perf_counter()
    img.save("test.png")
    #print("save took %i seconds"%(time.perf_counter() - start))

    start = time.perf_counter()
    boxes = od.inference_as_raw_output(np.array(img), to_file=True)
    print("inference_as_raw_output took %i seconds"%(time.perf_counter() - start))

    
    tree_x, tree_y = od.GetCenterObject(boxes, od.ObjectTypes.tree_index_h)
    if tree_x != -1:
        keyboard.send_keys("e", vk_packet=False)
        print("Harvest tree")
    tree_x, tree_y = od.GetCenterObject(boxes, od.ObjectTypes.tree_index)
    if tree_x != -1:
        tree_y -= 0.5
        tree_y *= height
        WindMouse.wind_mouse(np.round((tree_x-0.5)*width*0.3), 0)
        print("Aim to tree %s, %s"%(np.round((tree_x-0.5)*width), 0))

    print(boxes)
    time.sleep(10)







