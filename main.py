from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import cv2
from PIL import Image, ImageDraw
import numpy as np
from torchvision import transforms

def capture_patient_face():
    # Determine if an NVIDIA GPU is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))

    # Initialize the MTCNN model for face detection
    mtcnn = MTCNN(keep_all=True, device=device)

    # Initialize the InceptionResnetV1 model for face recognition
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    # Prompt for the name of the patient
    patient_name = input("Enter the name of the patient: ")

    # Open the webcam video capture
    video_capture = cv2.VideoCapture(0)

    # Preprocess for face image
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    try:
        while True:
            # Read a frame from the webcam
            ret, frame = video_capture.read()

            # Convert the frame to PIL Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)

            # Detect faces
            boxes, _ = mtcnn.detect(pil_image)

            # Draw faces on the frame
            draw = ImageDraw.Draw(pil_image)
            if boxes is not None:
                for box in boxes:
                    draw.rectangle(box.tolist(), outline=(255, 0, 0), width=6)

                    # Extract face image from the frame
                    face_image = pil_image.crop(box)
                    face_tensor = transform(face_image).unsqueeze(0).to(device)

                    # Generate face embedding using InceptionResnetV1
                    face_embedding = resnet(face_tensor)

                    # Perform face recognition inference here
                    # You can compare the face embedding with your known embeddings
                    # and predict the person's identity

                    # Save the whole picture with the bounding box
                    pil_image.save(f"{patient_name}.jpg")
                    break

            # Convert the PIL Image back to OpenCV format
            frame_drawn = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

            # Display the frame
            cv2.imshow('Webcam', frame_drawn)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass

    # Release the video capture and close the OpenCV windows
    video_capture.release()
    cv2.destroyAllWindows()

# Call the function to capture the patient's face
if __name__ == '__main__':
    capture_patient_face()