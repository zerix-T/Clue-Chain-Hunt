# Clue Hunt Solver

This package contains the two ROS 2 nodes used for the leader/follower clue-hunt task:

- `hunt_node.py`: leader logic
- `follower_node.py`: follower logic

## 1. Leader node (`hunt_node`)

The leader node is responsible for reading clue boards, validating the clue chain, navigating to target points, and finding the treasure.

Key responsibilities:

- Reads camera frames and camera calibration data
- Detects ArUco markers and QR clues from board images
- Validates clue tokens and command chain order
- Publishes valid clue data to:
  - `/hunt/clues`
  - `/hunt/boards`
  - `/hunt/treasure`
- Uses Nav2 to move to board, pillar, and treasure locations
- Uses LiDAR and colour-based pillar detection to resolve `PILLAR` and `BETWEEN` commands
- Publishes velocity commands to `/cmd_vel`

## 2. Follower node (`follower_node`)

The follower node tracks the leader and follows it while maintaining a safe gap.

Key responsibilities:

- Watches the follower camera for ArUco tag ID `49` on the leader
- Uses OpenCV + TF to estimate the leader position in the follower frame
- Keeps a trail of leader positions and follows them using a simple pursuit controller
- Publishes velocity commands to `/follower/cmd_vel`
- Falls back to a search motion if the leader tag is temporarily lost

## 3. How they work together

- The leader reads the world and solves the clue chain.
- The follower tracks the leader with its camera and follows at a fixed distance.
- Both nodes are launched together with:

```bash
ros2 launch clue_hunt_solver hunt.launch.py
```

## 4. Runtime notes

- `hunt_node` expects map and TF data from the navigation stack.
- `follower_node` relies on the leader tag being visible in the follower camera.
- The package depends on OpenCV, `cv_bridge`, TF2, Nav2, and ROS 2 geometry/message types.
