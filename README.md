# NMS_GPU
 NMS use cuda c implement.
 The C++ can use it with the .hpp file.
 In the NMS, decoding part is parallel wiht a batch,and the calculating iou is parallel with a frame. So the pipline is  YOLO->decoding->loop(batch calculate iou and nms).  
