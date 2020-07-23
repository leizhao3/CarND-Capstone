from styx_msgs.msg import TrafficLight
import numpy as np
import tensorflow as tf
import os
import cv2
import rospy
import time


LABELS = [TrafficLight.RED,TrafficLight.YELLOW,TrafficLight.GREEN,TrafficLight.UNKNOWN]
LABELS_NAME = ["RED", "YELLOW", "GREEN", "UNKNOWN"]

class TLClassifier(object):
    def __init__(self):
        model = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frozen_inference_graph.pb") #for classification (transfer learning)
        assert os.path.exists(model), "model file not found at [%s]" % (model)

        self.time_taken_for_inference = 0
        
        # For reading inference graph of pre trained model
        d_graph = tf.Graph()
        graph_def = tf.GraphDef()
        with open(model, 'rb') as f:
            graph_def.ParseFromString(f.read())
        with d_graph.as_default():
            tf.import_graph_def(graph_def)

        #Setting Holders
        self.image_tensor = d_graph.get_tensor_by_name("import/image_tensor:0")
        self.num_detections = d_graph.get_tensor_by_name("import/num_detections:0")
        self.detection_scores = d_graph.get_tensor_by_name("import/detection_scores:0")
        self.detection_boxes = d_graph.get_tensor_by_name("import/detection_boxes:0")
        self.detection_classes = d_graph.get_tensor_by_name("import/detection_classes:0")
        

        o_config = tf.ConfigProto()
        o_config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

        # Loading the classification graph
        self.session = tf.Session(graph=d_graph, config=o_config)

    def get_classification(self, image):
        """Determines the color of the traffic light in the image
        Args:
            image (cv::Mat): image containing the traffic light
        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        # Preprocessing Image
        img = cv2.resize(image, (300, 300))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)

        # Detecting Traffic Lights
        start = time.time()
        num_detections, classes, scores, boxes = self.session.run([self.num_detections, self.detection_classes, self.detection_scores, self.detection_boxes],
                                                  feed_dict={self.image_tensor: np.expand_dims(img, axis=0)})
        self.time_taken_for_inference = time.time() - start
        rospy.logdebug("Time taken for inference: %s" % (self.time_taken_for_inference))
        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes)
        num_detections = np.squeeze(num_detections).astype(np.uint32)

        # For identifying signal color
        for e in range(num_detections):
            class_idx = classes[e]

            if scores[e] > 0.50:
                rospy.loginfo("Identified Traffic Light: %s" % (LABELS_NAME[int(class_idx)]))
                return LABELS[int(class_idx)]

        return TrafficLight.UNKNOWN