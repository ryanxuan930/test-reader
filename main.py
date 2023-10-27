import cv2
import numpy as np
import pdf2image

# Convert PDF to image
images = pdf2image.convert_from_path('test.pdf')
img = np.array(images[0])

# Convert image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Canny edge detection
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# Find contours in the image
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter for rectangles
rectangles = []
for contour in contours:
    approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
    if len(approx) == 4:
        rectangles.append(approx)

# Find the largest rectangle
max_area = 0
max_rect = None
for rect in rectangles:
    area = cv2.contourArea(rect)
    if area > max_area:
        max_area = area
        max_rect = rect

# Draw the largest rectangle
cv2.drawContours(img, [max_rect], -1, (0, 0, 255), 2)

# Divide the rectangle into 32 rows and 22 columns
rows = np.linspace(max_rect[0][0][1], max_rect[2][0][1], num=33, dtype=int)
cols = np.linspace(max_rect[0][0][0], max_rect[2][0][0], num=23, dtype=int)

letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
for i in range(11, 22):
    for j in range(11, 23):
        x1, y1 = cols[j-1], rows[i-1]
        x2, y2 = cols[j], rows[i]
        cell = gray[y1:y2, x1:x2]
        black_pixels = np.count_nonzero(cell == 0)
        total_pixels = cell.shape[0] * cell.shape[1]
        if black_pixels / total_pixels >= 0.5:
            print(f"Cell ({letters[j-11]}, {i}) is mostly black.")
        else:
            print(f"Cell ({letters[j-11]}, {i}) is not mostly black.")

# Draw the grid on the image
for row in rows:
    cv2.line(img, (max_rect[0][0][0], row), (max_rect[2][0][0], row), (0, 255, 0), 1)
for col in cols:
    cv2.line(img, (col, max_rect[0][0][1]), (col, max_rect[2][0][1]), (0, 255, 0), 1)

# Display the image
cv2.imshow('Grid', img)
cv2.waitKey(0)
cv2.destroyAllWindows()