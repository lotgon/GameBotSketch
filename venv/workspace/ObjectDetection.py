import sys, time, os 
import tensorflow as tf # import tensorflow
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm
from enum import IntEnum

# importing all scripts that will be needed to export your model and use it for inference
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder


class ObjectDetection(object):
    """description of class"""
    class ObjectTypes(IntEnum):
        boar_index = 2
        tree_index = 7
        tree_index_h = 8


    def __init__(self, path2config="pipeline.config", path2modelCheckpoint="d:\\exported-models\\model1", 
                 path2label_map = 'tf_label_map.pbtxt ', \
                 cuda_device="0"):
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" # do not change anything in here
        # specify which device you want to work on.
        # Use "-1" to work on a CPU. Default value "0" stands for the 1st GPU that will be used
        os.environ["CUDA_VISIBLE_DEVICES"]=cuda_device # TODO: specify your computational device
        # checking that GPU is found
        if tf.test.gpu_device_name():
            print('GPU found')
        else:
            print("No GPU found")

        # do not change anything in this cell
        configs = config_util.get_configs_from_pipeline_file(path2config) # importing config
        model_config = configs['model'] # recreating model config``````
        self.detection_model = model_builder.build(model_config=model_config, is_training=False) # importing model

        ckpt = tf.compat.v2.train.Checkpoint(model=self.detection_model)
        chk_name = tf.train.latest_checkpoint(path2modelCheckpoint)
        #ckpt.restore(tf.train.latest_checkpoint(os.path.join(path2model, 'checkpoint'))).expect_partial()
        ckpt.restore(chk_name).expect_partial()

        self.category_index = label_map_util.create_category_index_from_labelmap(path2label_map,use_display_name=True)

    def __detect_fn(self, image):
        """
        Detect objects in image.
    
        Args:
          image: (tf.tensor): 4D input image
      
        Returs:
          detections (dict): predictions that model made
        """
        start = time.perf_counter()
        image, shapes = self.detection_model.preprocess(image)
        print("preprocess took %i seconds"%(time.perf_counter() - start))

        start = time.perf_counter()
        prediction_dict = self.detection_model.predict(image, shapes)
        print("predict took %i seconds"%(time.perf_counter() - start))

        start = time.perf_counter()
        detections = self.detection_model.postprocess(prediction_dict, shapes)
        print("postprocess took %i seconds"%(time.perf_counter() - start))

        return detections

    def __nms(self, rects, thd=0.5):
    #Filter rectangles
    #rects is array of oblects ([x1,y1,x2,y2], confidence, class)
    #thd - intersection threshold (intersection divides min square of rectange)
        out = []
        remove = [False] * len(rects)

        for i in range(0, len(rects) - 1):
            if remove[i]:
                continue
            inter = [0.0] * len(rects)
            for j in range(i, len(rects)):
                if remove[j]:
                    continue
                inter[j] = self.__intersection(rects[i][0], rects[j][0]) / min(self.__square(rects[i][0]), self.__square(rects[j][0]))

            max_prob = 0.0
            max_idx = 0
            for k in range(i, len(rects)):
                if inter[k] >= thd:
                    if rects[k][1] > max_prob:
                        max_prob = rects[k][1]
                        max_idx = k

            for k in range(i, len(rects)):
                if (inter[k] >= thd) & (k != max_idx):
                    remove[k] = True

        for k in range(0, len(rects)):
            if not remove[k]:
                out.append(rects[k])

        boxes = [box[0] for box in out]
        scores = [score[1] for score in out]
        classes = [cls[2] for cls in out]
        return boxes, scores, classes

    def __intersection(self, rect1, rect2):
        """
        Calculates square of intersection of two rectangles
        rect: list with coords of top-right and left-boom corners [x1,y1,x2,y2]
        return: square of intersection
        """
        x_overlap = max(0, min(rect1[2], rect2[2]) - max(rect1[0], rect2[0]));
        y_overlap = max(0, min(rect1[3], rect2[3]) - max(rect1[1], rect2[1]));
        overlapArea = x_overlap * y_overlap;
        return overlapArea


    def __square(self, rect):
        """
        Calculates square of rectangle
        """
        return abs(rect[2] - rect[0]) * abs(rect[3] - rect[1])

    def inference_as_raw_output(self, image_np,
                            box_th = 0.25,
                            nms_th = 0.5,
                            to_file = False):
        #Function that performs inference and return filtered predictions
        #Args:
          #box_th: (float) value that defines threshold for model prediction. Consider 0.25 as a value.
          #nms_th: (float) value that defines threshold for non-maximum suppression. Consider 0.5 as a value.
          #to_file: (boolean). When passed as True => results are saved into a file. Writing format is
          #path2image + (x1abs, y1abs, x2abs, y2abs, score, conf) for box in boxes
          #data: (str) name of the dataset you passed in (e.g. test/validation)
        #Returs:
          #detections (dict): filtered predictions that model made
        image_np = image_np[:, :, 0:3]
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
        detections = self.__detect_fn(input_tensor)
        
        # checking how many detections we got
        num_detections = int(detections.pop('num_detections'))
        
        # filtering out detection in order to get only the one that are indeed detections
        detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
        
        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
        
        # defining what we need from the resulting detection dict that we got from model output
        key_of_interest = ['detection_classes', 'detection_boxes', 'detection_scores']
        
        # filtering out detection dict in order to get only boxes, classes and scores
        detections = {key: value for key, value in detections.items() if key in key_of_interest}
        
        if box_th: # filtering detection if a confidence threshold for boxes was given as a parameter
            for key in key_of_interest:
                scores = detections['detection_scores']
                current_array = detections[key]
                filtered_current_array = current_array[scores > box_th]
                detections[key] = filtered_current_array
        
        if nms_th: # filtering rectangles if nms threshold was passed in as a parameter
            # creating a zip object that will contain model output info as
            output_info = list(zip(detections['detection_boxes'],
                                    detections['detection_scores'],
                                    detections['detection_classes']
                                    )
                                )
            boxes, scores, classes = self.__nms(output_info)
            
            detections['detection_boxes'] = boxes # format: [y1, x1, y2, x2]
            detections['detection_scores'] = scores
            detections['detection_classes'] = classes
            
        if to_file: # if saving to txt file was requested

            image_h, image_w, _ = image_np.shape
            file_name = f'pred_result.txt'
            
            line2write = list()
            
            with open(file_name, 'a+') as text_file:
                # iterating over boxes
                for b, s, c in zip(boxes, scores, classes):
                    
                    y1abs, x1abs = b[0] * image_h, b[1] * image_w
                    y2abs, x2abs = b[2] * image_h, b[3] * image_w
                    
                    list2append = [x1abs, y1abs, x2abs, y2abs, s, c]
                    line2append = ','.join([str(item) for item in list2append])
                    
                    line2write.append(line2append)
                
                line2write = ' '.join(line2write)
                text_file.write(line2write + os.linesep)
        
        return detections
    def GetCenterObject(self, boxes, object_type):
        #boxes = list of boxes
        #object_type  - looking for this type of object ObjectTypes
        # return center of the object
        object_i = boxes['detection_classes'].index(object_type.value) if object_type.value in boxes['detection_classes'] else -1
        if object_i >= 0:
            x1, y1, x2, y2 = boxes['detection_boxes'][object_i]
            print("x1=%s, y1=%s, x2=%s, y2=%s"%(x1, y1, x2, y2))
            return (x1+x2)/2, (y1+y2)/2
        return -1, -1



