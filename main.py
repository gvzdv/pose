from ultralytics import YOLO
import ai_gym
import cv2

model = YOLO("yolov8n-pose.pt")

# Change this line to use the default webcam
cap = cv2.VideoCapture(0)
assert cap.isOpened(), "Error accessing webcam"

w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# I turned off video recording but kept it in the code just in case
# video_writer = cv2.VideoWriter("workouts.avi",
#                                cv2.VideoWriter_fourcc(*'mp4v'),
#                                fps,
#                                (w, h))

gym_object = ai_gym.AIGym()
gym_object.set_args(line_thickness=2,
                    view_img=True,
                    pose_type="slouch",
                    kpts_to_check=[2, 6, 14],
                    slouch_threshold_angle=124)

tracking_enabled = True
end_of_session = False

frame_count = 0
while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Webcam frame is empty or webcam processing has been successfully completed.")
        break
    frame_count += 1
    results = model.predict(im0, verbose=False)

    if tracking_enabled:
        im0 = gym_object.start_counting(im0, results, frame_count, fps)
    # video_writer.write(im0)

    # Check for keypresses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Press 'q' to quit the program
        end_of_session = True
        break
    elif key == ord('s'):  # Press 's' to start/stop tracking
        tracking_enabled = not tracking_enabled  # Toggle the tracking_enabled flag
        print(f"Tracking is {'enabled' if tracking_enabled else 'disabled'}.")

# Add the code to handle the end of tracking here
if end_of_session:
    # Update the last durations for slouching or not slouching
    if gym_object.slouch_start_frame is not None:
        gym_object.total_slouch_time += (frame_count - gym_object.slouch_start_frame) / fps
    if gym_object.not_slouch_start_frame is not None:
        gym_object.total_not_slouch_time += (frame_count - gym_object.not_slouch_start_frame) / fps

    # Calculate the percentage of time not slouching
    total_time = gym_object.total_slouch_time + gym_object.total_not_slouch_time
    if total_time > 0:  # Avoid division by zero
        not_slouching_percentage = (gym_object.total_not_slouch_time / total_time) * 100
    else:
        not_slouching_percentage = 100  # If there's no time recorded, we'll assume 100% not slouching

    # Check the condition for the reward
    if total_time > 600 and not_slouching_percentage >= 85:
        reward_message = f"Congratulations! You've been not slouching for {round(not_slouching_percentage)}% of the time " \
                         f"({round(gym_object.total_not_slouch_time)} s.) and earned a cup of boba tea for good posture!"
    else:
        reward_message = f"Bad Mike! You slouched {round(100 - not_slouching_percentage)}% of the time ({round(gym_object.total_slouch_time)} s.). " \
                         f"Keep working on your posture to earn boba next time."

    print(reward_message)

cv2.destroyAllWindows()
# video_writer.release()
