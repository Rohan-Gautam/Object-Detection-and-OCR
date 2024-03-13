import cv2
import pytesseract

# Path to Tesseract executable (change it to match your system)
pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/bin/tesseract'  # Update with correct path

# Start webcam
cap = cv2.VideoCapture(0)

processing_image = False
current_text = ""

while True:
    # Capture frame from webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Display the frame
    if processing_image:
        try:
            # Convert frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Apply thresholding to preprocess the image
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Noise reduction using GaussianBlur
            denoised = cv2.GaussianBlur(thresh, (5, 5), 0)

            # Perform OCR
            text = pytesseract.image_to_string(denoised)
            print("OCR Result:", text)
            current_text = text
        except pytesseract.TesseractError as e:
            print("Tesseract Error:", e)

        # Reset processing_image flag
        processing_image = False

    # Overlay text on the frame if OCR has been performed
    if current_text:
        text_position = (10, 50)  # Adjust the position of the text
        text_color = (0, 0, 255)  # Red color
        cv2.putText(frame, current_text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)

    cv2.imshow('OCR', frame)

    # Check for key press events
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord(' '):  # Check for spacebar press
        processing_image = True

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
