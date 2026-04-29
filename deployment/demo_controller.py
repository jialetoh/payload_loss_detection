#!/usr/bin/env python
from __future__ import print_function

import argparse
import math
import os
import subprocess
import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Int8

class DemoController(object):
    def __init__(self, cmd_topic, odom_topic, speed, target_distance):
        self.cmd_topic = cmd_topic
        self.speed = speed
        self.target_distance = target_distance
        
        self.start_x = None
        self.start_y = None
        self.dist_traveled = 0.0
        self.loss_detected = False

        self.pub = rospy.Publisher(self.cmd_topic, Twist, queue_size=1)
        self.odom_sub = rospy.Subscriber(odom_topic, Odometry, self._on_odom, queue_size=1)
        
        # Subscribes to the alert published by your Siamese Node
        self.loss_sub = rospy.Subscriber('/payload/loss_detected', Int8, self._on_loss, queue_size=1)

    def _on_odom(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        if self.start_x is None:
            self.start_x = x
            self.start_y = y
        # Calculate straight-line distance traveled
        self.dist_traveled = math.hypot(x - self.start_x, y - self.start_y)

    def _on_loss(self, msg):
        if msg.data == 1:
            self.loss_detected = True

    def wait_for_subscriber(self, timeout_sec=3.0):
        deadline = rospy.Time.now() + rospy.Duration(timeout_sec)
        while not rospy.is_shutdown():
            if self.pub.get_num_connections() > 0:
                return
            if rospy.Time.now() >= deadline:
                return
            rospy.sleep(0.05)

    def stop_bot(self):
        stop_cmd = Twist()
        # Publish multiple 0-velocity commands to ensure the motor controller receives it
        for _ in range(10):
            self.pub.publish(stop_cmd)
            rospy.sleep(0.05)

    def run(self):
        self.wait_for_subscriber()
        if self.pub.get_num_connections() == 0:
            rospy.logwarn("No subscribers connected on %s, publishing anyway", self.cmd_topic)

        cmd = Twist()
        cmd.linear.x = self.speed
        rate = rospy.Rate(20.0)

        rospy.loginfo("Demo started. Target: %.1f meters at %.2f m/s", self.target_distance, self.speed)

        while not rospy.is_shutdown():
            if self.loss_detected:
                rospy.logwarn("PAYLOAD LOSS DETECTED! Executing emergency stop.")
                try:
                    subprocess.Popen(['aplay', os.path.expanduser('~/arte_robot/sounds/three_beeps.wav')])
                except Exception as e:
                    rospy.logwarn("Failed to play sound: %s", e)
                self.stop_bot()
                os._exit(0)

            if self.dist_traveled >= self.target_distance:
                rospy.loginfo("Successfully traveled %.2f meters. Stopping bot.", self.dist_traveled)
                self.stop_bot()
                os._exit(0)

            self.pub.publish(cmd)
            rate.sleep()

def parse_args():
    parser = argparse.ArgumentParser(description='Drive 5 meters, stop if payload loss detected.')
    parser.add_argument('--speed', type=float, default=0.4, help='Forward speed in m/s (default: 0.4)')
    parser.add_argument('--distance', type=float, default=5.0, help='Target distance in meters (default: 5.0)')
    parser.add_argument('--cmd-topic', default='/RosAria/cmd_vel', help='Command topic (default: /RosAria/cmd_vel)')
    parser.add_argument('--odom-topic', default='/odom', help='Odometry topic (default: /odom)')
    return parser.parse_args(rospy.myargv()[1:])

def main():
    rospy.init_node('payload_demo_controller', anonymous=False)
    args = parse_args()
    
    controller = DemoController(args.cmd_topic, args.odom_topic, args.speed, args.distance)
    controller.run()
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
