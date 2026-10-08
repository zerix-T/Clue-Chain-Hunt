#!/usr/bin/env python3
import hashlib
import math
import re
from collections import deque

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, Twist
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, LaserScan
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener

BOARD_APPROACH = 1.0       # m in front of every valid board
PILLAR_STANDOFF = 1.8      # m short of a pillar / target point
BETWEEN_STANDOFF = 0.6     # m, close approach to the first pillar of BETWEEN
PILLAR_RADIUS = 0.15       # m, LiDAR surface range -> pillar centre
GOTO_Y_OFFSET = 0.9        # m, added to the y of GOTO
ROT_MAX = 0.4              # rad/s
ROT_RAMP = 0.02            # rad/s per scan message
PILLAR_MIN_AREA = 300      # px^2
HSV_RANGES = {
    'RED':   [((0, 120, 60), (10, 255, 255)), ((170, 120, 60), (180, 255, 255))],
    'GREEN': [((40, 100, 50), (85, 255, 255))],
    'BLUE':  [((100, 100, 50), (130, 255, 255))],
}

QR_PATTERN = r"^HUNT:([^:]+):([^:]+):(TREASURE )?(.+)$"
QR_SIZE = 300
UNIT_SQUARE = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
QR_BOX = np.float32([[1.2, 0], [2.3, 0], [2.3, 1.0], [1.2, 1.0]])   # in marker widths
QR_DEST = np.float32([[0, 0], [QR_SIZE - 1, 0], [QR_SIZE - 1, QR_SIZE - 1], [0, QR_SIZE - 1]])
MARKER_PTS = np.float32([[-0.12, 0.12, 0], [0.12, 0.12, 0], [0.12, -0.12, 0], [-0.12, -0.12, 0]])


def chain_token(text):
    return hashlib.sha1(text.encode()).hexdigest()[:4].upper()


class HuntNode(Node):
    def __init__(self):
        super().__init__('hunt_node')

        self.bridge = CvBridge()
        self.aruco = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), cv2.aruco.DetectorParameters())
        self.qr_detector = cv2.QRCodeDetector()
        self.camera_matrix = None
        self.dist_coeffs = None
        self.rvec = self.tvec = None
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.prev_token = chain_token("START")
        self.cmd_queue = deque()       # (is_treasure, command, board)
        self.approach_goal = None      # 1 m in front of the last valid board
        self.navigating = False
        self.goal_treasure = False
        self.goal_pose = None
        self.done = False

        self.targets = None            # pillar colours being resolved
        self.between = None            # f of BETWEEN
        self.pillars = {}              # colour -> map xy
        self.pillar_color = None
        self.pillar_bearing = None
        self.ang_vel = 0.0

        self.create_subscription(Image, '/camera/image_raw', self.on_image, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, '/camera/camera_info', self.on_info, qos_profile_sensor_data)
        self.create_subscription(LaserScan, '/scan', self.on_scan, qos_profile_sensor_data)
        self.clue_pub = self.create_publisher(String, '/hunt/clues', 10)
        self.board_pub = self.create_publisher(String, '/hunt/boards', 10)
        self.treasure_pub = self.create_publisher(PoseStamped, '/hunt/treasure', 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.nav = ActionClient(self, NavigateToPose, 'navigate_to_pose')

    # ---------------------------------------------------------------- geometry
    def lookup_map_T(self, frame):
        """(R, t) of `frame` in the map frame."""
        T = self.tf_buffer.lookup_transform("map", frame, Time()).transform
        x, y, z, w = T.rotation.x, T.rotation.y, T.rotation.z, T.rotation.w
        R = np.array([[1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
                      [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
                      [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]])
        return R, np.array([T.translation.x, T.translation.y, T.translation.z])

    def get_qr_from_aruco(self, frame, corners):
        H = cv2.getPerspectiveTransform(UNIT_SQUARE, np.float32(corners))
        qr_pts = cv2.perspectiveTransform(QR_BOX[None], H)[0]
        qr = cv2.warpPerspective(frame, cv2.getPerspectiveTransform(qr_pts, QR_DEST), (QR_SIZE, QR_SIZE))
        return qr, qr_pts

    def pub_aruco_pose(self, m_id, img_pts):
        """Publish the board centre; return (centre xy, face normal xy) in the map frame."""
        ok, self.rvec, self.tvec = cv2.solvePnP(
            MARKER_PTS, img_pts, self.camera_matrix, self.dist_coeffs, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not ok:
            return None
        try:
            R, t = self.lookup_map_T("cam_optical_link")
        except Exception as e:
            self.get_logger().warn(f"TF failed: {e}")
            return None

        p = R @ self.tvec.ravel() + t
        self.board_pub.publish(String(data=f"{m_id} {p[0]:.3f} {p[1]:.3f}"))

        n = (R @ cv2.Rodrigues(self.rvec)[0][:, 2])[:2]     # marker z = out of the face
        n = n / np.linalg.norm(n)
        if n @ (t[:2] - p[:2]) < 0:                         # make it point towards the camera
            n = -n
        return p[:2], n

    def near(self, target, robot, standoff=PILLAR_STANDOFF):
        """Pose `standoff` m short of target, facing it."""
        d = target - robot
        dist = np.linalg.norm(d)
        d = d / dist
        g = target - min(standoff, dist) * d
        return g[0], g[1], math.atan2(d[1], d[0])

    # ---------------------------------------------------------------- navigation
    def go_to(self, x, y, yaw=0.0):
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = 'map'
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = float(x)
        goal.pose.pose.position.y = float(y)
        goal.pose.pose.orientation.z = math.sin(yaw / 2.0)
        goal.pose.pose.orientation.w = math.cos(yaw / 2.0)

        self.goal_pose = goal.pose
        self.navigating = True
        self.ang_vel = 0.0
        self.nav.send_goal_async(goal).add_done_callback(self.goal_response)

    def goal_response(self, future):
        handle = future.result()
        if not handle.accepted:
            self.get_logger().warn('Goal rejected')
            self.navigating = False
            return
        handle.get_result_async().add_done_callback(self.goal_done)

    def goal_done(self, future):
        self.navigating = False
        status = future.result().status
        if status != 4:
            self.get_logger().warn(f'Navigation ended: {status}')
        elif self.goal_treasure:
            self.treasure_pub.publish(self.goal_pose)
            self.done = True
            self.get_logger().info('Treasure reached - mission success!')

    def rotate(self):
        self.ang_vel = min(self.ang_vel + ROT_RAMP, ROT_MAX)
        twist = Twist()
        twist.angular.z = self.ang_vel
        self.cmd_pub.publish(twist)

    def stop(self):
        self.ang_vel = 0.0
        self.cmd_pub.publish(Twist())

    # ---------------------------------------------------------------- pillars
    def detect_pillar(self, frame):
        """Store the bearing of the largest wanted-colour blob, if any."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        best_area, best_color, best_c = PILLAR_MIN_AREA, None, None
        for color in self.targets:
            if color in self.pillars:
                continue
            mask = np.zeros(hsv.shape[:2], np.uint8)
            for lo, hi in HSV_RANGES[color]:
                mask |= cv2.inRange(hsv, lo, hi)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            c = max(contours, key=cv2.contourArea, default=None)
            if c is not None and cv2.contourArea(c) >= best_area:
                best_area, best_color, best_c = cv2.contourArea(c), color, c

        if best_color is None:
            return
        x, _, w, _ = cv2.boundingRect(best_c)
        self.pillar_color = best_color
        self.pillar_bearing = -math.atan2(x + w / 2.0 - self.camera_matrix[0, 2], self.camera_matrix[0, 0])

    def pillar_pos(self, scan):
        """Pillar centre (map xy) at the stored bearing, and the robot xy."""
        n = len(scan.ranges)
        idx = int(round(((self.pillar_bearing - scan.angle_min) % (2 * math.pi)) / scan.angle_increment))
        half = max(1, int(math.radians(3.0) / scan.angle_increment))
        rs = [scan.ranges[(idx + k) % n] for k in range(-half, half + 1)]
        rs = [r for r in rs if math.isfinite(r) and scan.range_min < r < scan.range_max]
        if not rs:
            return None
        r, b = min(rs) + PILLAR_RADIUS, self.pillar_bearing
        R, t = self.lookup_map_T("lidar_link")
        return (R @ np.array([r * math.cos(b), r * math.sin(b), 0.0]) + t)[:2], t[:2]

    def locate_pillar(self, scan):
        self.stop()
        res = self.pillar_pos(scan)
        self.pillar_bearing = None
        if res is None:
            return
        pillar, robot = res
        self.pillars[self.pillar_color] = pillar
        self.get_logger().info(f"Saved {self.pillar_color} at {np.round(pillar, 2)}")

        if self.between is None:                                    # PILLAR
            self.targets = None
            self.go_to(*self.near(pillar, robot))
        elif len(self.pillars) == 2:                                # BETWEEN, both found
            p1, p2 = (self.pillars[c] for c in self.targets)
            self.targets = None
            self.go_to(*self.near(p1 + self.between * (p2 - p1), robot))
        else:                                                       # BETWEEN, first pillar
            self.go_to(*self.near(pillar, robot, BETWEEN_STANDOFF))

    # ---------------------------------------------------------------- commands
    def start_command(self, treasure, command, board):
        """Run a clue command. Returns True if a pillar search was started."""
        self.goal_treasure = treasure
        cmd, *args = command.split()
        if cmd == "GOTO":
            self.go_to(float(args[0]), float(args[1]) + GOTO_Y_OFFSET)
        elif cmd == "PILLAR":
            self.targets, self.between = [args[0].upper()], None
        elif cmd == "BETWEEN":
            self.targets, self.between = [args[0].upper(), args[1].upper()], float(args[2])
        elif cmd == "REL" and board is not None:
            a, b = float(args[0]), float(args[1])
            x_axis = board[1]                                       # out of the face
            y_axis = np.array([-x_axis[1], x_axis[0]])              # reader's right
            point = board[0] + a * x_axis + b * y_axis
            if treasure:
                self.go_to(*point)
            else:
                self.go_to(*self.near(point, self.lookup_map_T("lidar_link")[1][:2]))
        else:
            self.get_logger().warn(f"Command not implemented: {command}")

        if self.targets is None:
            return False
        self.pillars, self.pillar_bearing, self.ang_vel = {}, None, 0.0
        return True

    # ---------------------------------------------------------------- callbacks
    def accept_clue(self, data, treasure, command, marker_id, img_pts):
        self.prev_token = chain_token(data)
        self.clue_pub.publish(String(data=data))
        board = self.pub_aruco_pose(marker_id, img_pts)
        if board is not None and not treasure:
            p, n = board
            g = p + BOARD_APPROACH * n
            self.approach_goal = (g[0], g[1], math.atan2(-n[1], -n[0]))
        self.cmd_queue.append((bool(treasure), command, board))
        self.targets = None                                         # new board -> drop pillar search
        self.get_logger().info(f"VALID clue, queued: {command}")
        return board

    def on_image(self, msg):
        if self.camera_matrix is None:
            return
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        corners, ids, _ = self.aruco.detectMarkers(frame)

        for marker_corners, marker_id in zip(corners, ids if ids is not None else []):
            marker_id = int(np.asarray(marker_id).ravel()[0])
            img_pts = marker_corners[0]
            qr, qr_pts = self.get_qr_from_aruco(frame, img_pts)
            try:
                data, _, _ = self.qr_detector.detectAndDecode(qr)
            except cv2.error:
                data = ""
            m = re.match(QR_PATTERN, data) if data else None
            if m and m.group(2) == self.prev_token:
                board = self.accept_clue(data, m.group(3), m.group(4), marker_id, img_pts)
                if board is not None:
                    cv2.drawFrameAxes(frame, self.camera_matrix, self.dist_coeffs, self.rvec, self.tvec, 0.10)
            cv2.polylines(frame, [qr_pts.astype(np.int32)], True, (0, 255, 0), 2)
            cv2.aruco.drawDetectedMarkers(frame, [marker_corners], np.array([marker_id]))

        if self.targets and self.pillar_bearing is None:
            self.detect_pillar(frame)

        cv2.imshow("hunt cam", frame)
        cv2.waitKey(1)

    def on_info(self, msg):
        self.camera_matrix = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        self.dist_coeffs = np.array(msg.d, dtype=np.float64)

    def on_scan(self, scan):
        if self.done or self.navigating or not self.nav.server_is_ready():
            return

        if self.approach_goal is not None:
            self.goal_treasure = False
            self.go_to(*self.approach_goal)
            self.approach_goal = None
            return

        if self.targets is None:
            if not self.cmd_queue:
                self.rotate()                                       # search for the next board
                return
            if not self.start_command(*self.cmd_queue.popleft()):
                return

        if self.pillar_bearing is None:
            self.rotate()                                           # search for the pillar
        else:
            self.locate_pillar(scan)


def main():
    rclpy.init()
    node = HuntNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()