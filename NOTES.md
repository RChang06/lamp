# working notes (not the submission)

## setup findings
- python 3.12 via uv (mediapipe has no 3.14 wheels)
- mujoco loads the urdf directly, 5 joints, limits read from the file
- urdf import needs `discardvisual="false" fusestatic="false"` or mujoco drops the visual meshes and merges the fixed head/camera/light frames
- supplied lamp_shade.stl is ascii, mujoco only reads binary stl, converted to robot/assets/lamp_shade_bin.stl (original kept)
- visual geoms are group 1, collision geoms group 0, render with group 0 hidden
- head_pitch_joint has rpy yaw pi, so at zero pose the lamp head faces -x of the base, keep in mind for look-at math

## measured on dev laptop (windows, i7 cpu only)
- mediapipe face landmarker: ~10 ms/frame, ~30 fps from webcam
- piper tts (lessac medium): 0.29 s to make 3.3 s of speech
- faster-whisper base.en int8: 0.66 s to transcribe 3.3 s of speech, exact text

## open decisions
- cloud model for vision + planning (no api key set yet)
- brain/body protocol (proposed: json over websocket)
- how to split the work

## servo / physics findings
- first servos (kp=40) sagged under gravity, shoulder overshot into its joint limit and bounced, now kp = effort/0.05 rad, kv = effort/0.5, implicitfast integrator, 0.002 armature
- torque stays capped at the urdf effort limits (6, 12, 9, 3.5, 3.2 Nm)
- supplied head collision cylinder (r=0.11) overlaps the upper arm already at the zero pose, which jammed head pitch at 0, excluded self collision between upper arm, neck, gimbal, head
- every joint now reaches both ends of its range alone
- leaning forward (shoulder -0.68) with elbow folded (-1.67) saturates shoulder and elbow torque and sags 0.2 rad, a real reachability limit for the motion layer to avoid
