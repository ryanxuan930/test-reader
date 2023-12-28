import cv2
import numpy as np
from pdf2image import convert_from_path
import csv

with open('answer.csv', 'r') as csvfile:
    reader = list(csv.reader(csvfile))

# Parameters
PADDING = 12
SHRINK_SIZE = 10

pages = convert_from_path('input.pdf')

def main(page: int):
    SHEET = 0
    STUDENT_ID = ''
    ANSWER = ['' for i in range(60)]
    ERROR = ['' for i in range(60)]
    SCORE = np.zeros(60, dtype=float)
    BLACK_THRESHOLD = 160
    img = np.array(pages[page])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rectangles = []
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
        if len(approx) == 4:
            rectangles.append(approx)
    max_area = 0
    max_rect = None
    for rect in rectangles:
        area = cv2.contourArea(rect)
        if area > max_area:
            max_area = area
            max_rect = rect
    center = np.mean(max_rect, axis=0)
    shrinked_rect = []
    for point in max_rect:
        vector = point - center
        normalized_vector = vector / np.linalg.norm(vector)
        new_point = point - SHRINK_SIZE * normalized_vector
        shrinked_rect.append(new_point)
    shrinked_rect = np.array(shrinked_rect).astype(int)
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
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rows = np.linspace(0, height, num=33, dtype=int)
    cols = np.linspace(0, width, num=23, dtype=int)
    letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    test_type = np.zeros(10, dtype=int)
    for j in range(12, 22):
        x1, y1 = cols[j-1] + PADDING, rows[10] + PADDING
        x2, y2 = cols[j] - PADDING, rows[11] - PADDING
        cell = gray[y1:y2, x1:x2]
        total_pixels = cell.shape[0] * cell.shape[1]
        black_pixels = np.sum(cell < BLACK_THRESHOLD)
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
            temp[i-1] = black_pixels
        STUDENT_ID += id_numbers[np.argmax(temp)]
    answer_temp = np.zeros((60, 6), dtype=float)
    for i in range(12, 32):
        for j in range(0, 6):
            for k in range(0, 3):
                x1, y1 = cols[j+k*7+1] + PADDING, rows[i-1] + PADDING
                x2, y2 = cols[j+k*7+2] - PADDING, rows[i] - PADDING
                cell = gray[y1:y2, x1:x2]
                total_pixels = cell.shape[0] * cell.shape[1]
                black_pixels = np.sum(cell < BLACK_THRESHOLD)
                if total_pixels == 0:
                    answer_temp[i-12+k*20][j] = 0
                else:
                    answer_temp[i-12+k*20][j] = black_pixels/total_pixels
    for i in range(0, 60):
        if np.sum(answer_temp[i] > 0.3) > 0:
            for j in np.where(answer_temp[i] > 0.3)[0]:
                ANSWER[i] += letters[j]
            if ANSWER[i] == reader[7 + int(SHEET + 1)][i+1]:
                ERROR[i] = '='
                if reader[1][i+1] == '0':
                    SCORE[i] = reader[2][i+1]
                else:
                    temp_answer = list(ANSWER[i])
                    temp_reader = list(reader[7 + int(SHEET + 1)][i+1])
                    # Get union of two lists (ANSWER and reader) and subtract the intersection
                    temp = list(set(temp_answer) | set(temp_reader) - (set(temp_answer) & set(temp_reader)))
                    if len(temp) == 0:
                        ERROR[i] = '='
                        SCORE[i] = reader[2][i+1]
                    else:
                        ERROR[i] = ''.join(temp)
                        SCORE[i] = SCORE[2+len(temp)] = reader[2][i+1]
                    
            else:
                ERROR[i] = ANSWER[i]
        else:
            ANSWER[i] = 'X'
    print('Sheet: '+letters[SHEET], 'Student ID: '+STUDENT_ID, 'Total: ' + str(np.sum(SCORE)))
    print('Error: ' + ''.join(ERROR))
    return [page + 1, letters[SHEET], STUDENT_ID, np.sum(SCORE)] + ERROR

result_list = []
for i in range(0, len(pages)):
    print('Page: ' + str(i))
    result_list.append(main(i))
    print('---------------------------')
# Sort the result list by the score (index 3)
sorted_list = sorted(result_list, key=lambda x: x[3], reverse=True)
for i, row in enumerate(sorted_list):
    # same rank if the score is the same
    row.append(i+1)
    if i > 0 and row[3] == sorted_list[i-1][3]:
        row[-1] = sorted_list[i-1][-1]

with open('result.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['No.', 'Sheet', 'Student ID', 'Score'] + ['Q' + str(i+1) for i in range(60)] + ['Rank'])
    for row in sorted_list:
        writer.writerow(row)