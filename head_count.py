import cv2

# Initialize person counter
person_count = 0

# Load pre-trained person detection classifier (Haar Cascade)
classifier = cv2.CascadeClassifier("haarcascade_fullbody.xml")

# Initialize video capture from a camera or video file
cap = cv2.VideoCapture(0)  # You can replace 0 with the video file path

while True:
    # Read a frame from the video source
    ret, frame = cap.read()

    if not ret:
        break

    # Convert the frame to grayscale for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect persons in the frame
    persons = classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around detected persons
    for (x, y, w, h) in persons:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Update the person count
    person_count = len(persons)

    # Display the current person count
    cv2.putText(frame, f"Persons: {person_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow("Person Counter", frame)

    # Print the person count
    print(person_count)
    # Exit the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()