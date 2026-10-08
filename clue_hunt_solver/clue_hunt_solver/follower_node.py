#!/usr/bin/env python3
import math
from collections import deque

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped, Twist
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from tf2_geometry_msgs import do_transform_point
from tf2_ros import Buffer, TransformListener

TAG_ID = 49
H = 0.06                       # half tag side (m)
OBJ_PTS = np.float32([[-H, H, 0], [H, H, 0], [H, -H, 0], [-H, -H, 0]])
GAP = 0.8                      # desired distance to the leader (m)
LOOKAHEAD = 1.0                # trail point we aim at is at least this far (m)
STEP = 0.1                     # trail point spacing (m)
V_MAX, W_MAX = 0.5, 1.2
K_V, K_W = 0.8, 1.5
SEARCH_W = 0.4                 # rad/s when the tag is lost
LOST_AFTER = 0.5               # s


class FollowerNode(Node):
    def __init__(self):
        super().__init__('follower_node')
        self.bridge = CvBridge()
        self.detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), cv2.aruco.DetectorParameters())
        self.K = self.D = None
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.trail = deque(maxlen=300)   # leader positions in follower/odom
        self.leader = None
        self.seen = 0.0
        self.side = 1.0                  # +1: leader was last seen on the left

        self.create_subscription(Image, '/follower/camera/image_raw', self.on_image, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, '/follower/camera/camera_info', self.on_info, qos_profile_sensor_data)
        self.cmd_pub = self.create_publisher(Twist, '/follower/cmd_vel', 10)
        self.create_timer(0.05, self.control)

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def on_info(self, msg):
        self.K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        self.D = np.array(msg.d, dtype=np.float64)

    def on_image(self, msg):
        if self.K is None:
            return
        corners, ids, _ = self.detector.detectMarkers(self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8'))
        if ids is None or TAG_ID not in ids.ravel():
            return
        c = corners[list(ids.ravel()).index(TAG_ID)][0]
        ok, _, tvec = cv2.solvePnP(OBJ_PTS, c, self.K, self.D, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not ok:
            return
        try:
            tf = self.tf_buffer.lookup_transform('follower/odom', 'follower/cam_optical_link', Time())
        except Exception:
            return

        t = tvec.ravel()
        pt = PointStamped()
        pt.point.x, pt.point.y, pt.point.z = map(float, t)
        p = do_transform_point(pt, tf).point
        p = np.array([p.x, p.y])

        self.leader, self.seen = p, self.now()
        self.side = 1.0 if t[0] < 0 else -1.0
        if not self.trail or np.linalg.norm(p - self.trail[-1]) > STEP:
            self.trail.append(p)

    def control(self):
        try:
            T = self.tf_buffer.lookup_transform('follower/odom', 'follower/base_footprint', Time()).transform
        except Exception:
            return
        q = T.rotation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        pos = np.array([T.translation.x, T.translation.y])
        cmd = Twist()

        if self.leader is None:
            cmd.angular.z = SEARCH_W
            self.cmd_pub.publish(cmd)
            return

        while len(self.trail) > 1 and np.linalg.norm(self.trail[0] - pos) < LOOKAHEAD:
            self.trail.popleft()                                  # points already passed
        dx, dy = self.trail[0] - pos
        err = math.atan2(math.sin(math.atan2(dy, dx) - yaw), math.cos(math.atan2(dy, dx) - yaw))
        gap = np.linalg.norm(self.leader - pos)

        v = float(np.clip(K_V * (gap - GAP), 0.0, V_MAX)) * max(0.0, math.cos(err))
        w = float(np.clip(K_W * err, -W_MAX, W_MAX))
        if self.now() - self.seen > LOST_AFTER and v < 0.02 and abs(err) < 0.3:
            w = SEARCH_W * self.side                              # at last known spot: look around

        cmd.linear.x, cmd.angular.z = v, w
        self.cmd_pub.publish(cmd)


def main():
    rclpy.init()
    node = FollowerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cmd_pub.publish(Twist())
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()