import cv2
import numpy as np
from telegram import Bot
import asyncio

# Load YOLO model
def load_yolo():
    net = cv2.dnn.readNet(r"C:\Users\LENOVO\Desktop\Yolo\yolov4.weights", r"C:\Users\LENOVO\Desktop\Yolo\yolov4.cfg")
    with open(r"C:\Users\LENOVO\Desktop\Yolo\coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return net, classes, output_layers

# Detect people in a frame
def detect_crowd(net, output_layers, frame, threshold=10):
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > 0.5 and class_id == 0:  # Class ID 0 is "person" in YOLO
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    people_count = len(indexes)
    
    # Draw bounding boxes on the frame
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            color = (0, 255, 0)  # Green bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f'Person {i+1}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return people_count > threshold, people_count, frame  # Returning the modified frame with bounding boxes


# Send a screenshot via Telegram
async def send_screenshot_to_telegram(image_path, bot_token, chat_id):
    bot = Bot(token=bot_token)
    async with bot:
        with open(image_path, 'rb') as photo:
            await bot.send_photo(chat_id=chat_id, photo=photo)
            print(f"Screenshot sent to Telegram chat ID: {chat_id}")

# Main function
def process_video(video_path, bot_token, chat_id):
    net, classes, output_layers = load_yolo()
    cap = cv2.VideoCapture(video_path)

    frame_skip = 20  # Process every 20th frame
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        is_crowd, people_count, frame_with_boxes = detect_crowd(net, output_layers, frame, threshold=10)

        if is_crowd:
            print(f"Crowd detected! People count: {people_count}")
            screenshot_path = "screenshot.png"
            cv2.imwrite(screenshot_path, frame_with_boxes)  # Save the frame with bounding boxes

            # Send screenshot asynchronously
            asyncio.run(send_screenshot_to_telegram(screenshot_path, bot_token, chat_id))
            break  # Stop after the first detection, or remove this to continue detecting

        # Display the video feed with bounding boxes (optional, for debugging)
        cv2.imshow("Video Feed", frame_with_boxes)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Replace with your Telegram bot token and chat ID
BOT_TOKEN = ""
CHAT_ID = "1125896620"  # Replace with your chat ID or group's chat ID
VIDEO_PATH = r"C:\Users\LENOVO\Desktop\Telegram\Walking.mp4"
# Run the program
process_video(VIDEO_PATH, BOT_TOKEN, CHAT_ID)
