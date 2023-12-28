import cv2
import numpy as np
from pdf2image import convert_from_path
import csv
import tkinter as tk
from tkinter import filedialog
'''
def load_file():
    # Open a file dialog to select a file
    file_path = filedialog.askopenfilename()

    # Load the selected file
    with open(file_path, 'r') as csvfile:
        # Create a CSV reader object
        global reader
        reader = list(csv.reader(csvfile))
        label.config(text='Selected file: ' + file_path)

window = tk.Tk()
window.title('讀卡程式')
window.geometry('500x400')
label = tk.Label(window, text='No file selected')
label.pack()
button = tk.Button(window, text='選擇答案', command=load_file)
button.pack()
window.mainloop()
'''

with open('answer.csv', 'r') as csvfile:
    # Create a CSV reader object
    reader = list(csv.reader(csvfile))
# Result
SHEET = 0
STUDENT_ID = ''
ANSWER = ['' for i in range(60)]
ERROR = ['' for i in range(60)]
SCORE = np.zeros(60, dtype=float)
BLACK_THRESHOLD = 160
ANSWER_THRESHOLD = 0.25

# Parameters
PADDING = 12
SHRINK_SIZE = 10

# Load the PDF and convert to an image
pages = convert_from_path('DOC0024.pdf')
img = np.array(pages[0])

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply edge detection
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# Find the outer contours
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

# Shrink the rectangle by 5 pixels
center = np.mean(max_rect, axis=0)
shrinked_rect = []
for point in max_rect:
    vector = point - center
    normalized_vector = vector / np.linalg.norm(vector)
    new_point = point - SHRINK_SIZE * normalized_vector
    shrinked_rect.append(new_point)
shrinked_rect = np.array(shrinked_rect).astype(int)

# Rotate the largest rectangle to be horizontal
rect = cv2.minAreaRect(shrinked_rect)
box = cv2.boxPoints(rect)
box = np.int0(box)
width = int(rect[1][0])
height = int(rect[1][1])
src_pts = box.astype("float32")
dst_pts = np.array([[0, height-1],
                    [0, 0],
                    [width-1, 0],
                    [width-1, height-1]], dtype="float32")
M = cv2.getPerspectiveTransform(src_pts, dst_pts)
img = cv2.warpPerspective(img, M, (width, height))
cv2.imwrite("result2.png", img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# Divide the rectangle into 32 rows and 22 columns
rows = np.linspace(0, height, num=33, dtype=int)
cols = np.linspace(0, width, num=23, dtype=int)

# Draw the grid on the image
for row in rows:
    cv2.line(img, (cols[0], row), (cols[-1], row), (0, 255, 0), 1)
for col in cols:
    cv2.line(img, (col, rows[0]), (col, rows[-1]), (0, 255, 0), 1)
# Check each cell for black pixels
letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
test_type = np.zeros(10, dtype=int)
for j in range(12, 22):
    x1, y1 = cols[j-1] + PADDING, rows[10] + PADDING
    x2, y2 = cols[j] - PADDING, rows[11] - PADDING
    cell = gray[y1:y2, x1:x2]
    total_pixels = cell.shape[0] * cell.shape[1]
    black_pixels = np.sum(cell < BLACK_THRESHOLD)
    cv2.putText(img, str(round(black_pixels/total_pixels, 2)), (x1, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)
    test_type[j-12] = black_pixels
SHEET = int(np.argmax(test_type))

id_letters = ['A', 'B', 'D', 'I', 'M', 'N']
first_letter = np.zeros(6, dtype=int)
for i in range(1, 7):
    x1, y1 = cols[11] + PADDING, rows[i-1] + PADDING
    x2, y2 = cols[12] - PADDING, rows[i] - PADDING
    cell = gray[y1:y2, x1:x2]
    total_pixels = cell.shape[0] * cell.shape[1]
    black_pixels = np.sum(cell < BLACK_THRESHOLD)
    cv2.putText(img, str(round(black_pixels/total_pixels, 2)), (x1, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)
    first_letter[i-1] = black_pixels
STUDENT_ID += id_letters[np.argmax(first_letter)]

id_numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
for j in range(13, 22):
    temp = np.zeros(10, dtype=int)
    for i in range(1, 11):
        x1, y1 = cols[j-1] + PADDING, rows[i-1] + PADDING
        x2, y2 = cols[j] - PADDING, rows[i] - PADDING
        cell = gray[y1:y2, x1:x2]
        total_pixels = cell.shape[0] * cell.shape[1]
        black_pixels = np.sum(cell < BLACK_THRESHOLD)
        # Draw the cell
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
        cv2.putText(img, str(round(black_pixels/total_pixels, 2)), (x1, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)
        temp[i-1] = black_pixels
    STUDENT_ID += id_numbers[np.argmax(temp)]
# Answer
answer_temp = np.zeros((60, 6), dtype=float)
for i in range(12, 32):
    for j in range(0, 6):
        for k in range(0, 3):
            x1, y1 = cols[j+k*7+1] + PADDING, rows[i-1] + PADDING
            x2, y2 = cols[j+k*7+2] - PADDING, rows[i] - PADDING
            cell = gray[y1:y2, x1:x2]
            total_pixels = cell.shape[0] * cell.shape[1]
            black_pixels = np.sum(cell < BLACK_THRESHOLD)
            answer_temp[i-12+k*20][j] = round(black_pixels/total_pixels, 2)
            # Draw the cell
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
            cv2.putText(img, str(round(black_pixels/total_pixels, 2)), (x1, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)
for i in range(0, 60):
    if np.sum(answer_temp[i] > ANSWER_THRESHOLD) > 0:
        for j in np.where(answer_temp[i] > ANSWER_THRESHOLD)[0]:
            ANSWER[i] += letters[j]
        if ANSWER[i] == reader[7 + int(SHEET + 1)][i+1]:
            ERROR[i] = '='
            if reader[1][i+1] == '0':
                SCORE[i] = reader[2][i+1]
        else:
            ERROR[i] = ANSWER[i]
    else:
        ANSWER[i] = 'X'
print('Sheet: '+letters[SHEET], 'Student ID: '+STUDENT_ID, 'Total: ' + str(np.sum(SCORE)))
print('Answer: ' + str(ANSWER))
print('Error: ' + ''.join(ERROR))
print('Score: ' + str(SCORE))
cv2.imwrite("result.png", img)