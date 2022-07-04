Sensor Viewer
- Prints sensor predictions when sensor is attached and model has been pretrained.
- Ensure model name and COM port is set appropriately in file
- Ensure 5X_burst_stream.ino (Arduino Scripts Folder) is uploaded to arduino and is sending 20 datapoints
- Further parameters can be adjusted in setup section of file


Model Trainer
- Uses data collected by 3D Printer to train MLP


Sensor Tester
- Tests sensor & model accuracy with 3D printer


3D Data Collector
- Collects sample data for training
