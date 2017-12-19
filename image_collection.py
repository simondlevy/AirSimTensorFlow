#!/usr/bin/env python3
'''
image_collection.py : uses AirSim to collect vehicle first-person-view images

Copyright (C) 2017 Jack Baird, Alex Cantrell, Keith Denning, Rajwol Joshi, 
Simon D. Levy, Will McMurtry, Jacob Rosen

This file is part of AirSimTensorFlow

MIT License
'''

from AirSimClient import CarClient, CarControls, ImageRequest, AirSimImageType, AirSimClientBase
from image_helper import IMAGEDIR
import pprint
import os
import time

# We maintain a queue of images of this size
QUEUESIZE = 10

# Create image directory if it doesn't already exist
try:
    os.stat(IMAGEDIR)
except:
    os.mkdir(IMAGEDIR)
    
# connect to the AirSim simulator 
client = CarClient()
client.confirmConnection()
print('Connected')
client.enableApiControl(True)
car_controls = CarControls()

client.reset()

# go forward
car_controls.throttle = 1.0
car_controls.steering = 0
client.setCarControls(car_controls)

imagequeue = []

while True:

    # get RGBA camera images from the car
    responses = client.simGetImages([ImageRequest(1, AirSimImageType.Scene)])  

    # add image to queue        
    imagequeue.append(responses[0].image_data_uint8)

    # dump queue when it gets full
    if len(imagequeue) == QUEUESIZE:
        for i in range(QUEUESIZE):
            AirSimClientBase.write_file(os.path.normpath(IMAGEDIR + '/image%03d.png'  % i ), imagequeue[i])
        imagequeue.pop(0)    

    collision_info = client.getCollisionInfo()

    if collision_info.has_collided:
        print("Collision at pos %s, normal %s, impact pt %s, penetration %f, name %s, obj id %d" % (
            pprint.pformat(collision_info.position), 
            pprint.pformat(collision_info.normal), 
            pprint.pformat(collision_info.impact_point), 
            collision_info.penetration_depth, collision_info.object_name, collision_info.object_id))
        break

    time.sleep(0.1)

client.enableApiControl(False)
