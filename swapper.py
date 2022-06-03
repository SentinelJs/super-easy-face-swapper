import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

def detector(image, idx):
    points = []

    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    with mp_face_mesh.FaceMesh(
    static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5) as face_mesh:
        
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Print and draw face mesh landmarks on the image.
        if not results.multi_face_landmarks:
            return points
        annotated_image = image.copy()
        face_landmarks = results.multi_face_landmarks[0]
        for i in range(len(face_landmarks.landmark)):
            data_point = face_landmarks.landmark[i]
            if data_point.x > 1:
                data_point.x = 0.99
            if data_point.y > 1:
                data_point.y = 0.99
            points.append(np.array((
                    data_point.x*annotated_image.shape[0],
                    data_point.y*annotated_image.shape[1]
            )))

    points = np.array(points, np.int32)

    return points

def extract_index_nparray(nparray):
    index = None
    for num in nparray[0]:
        index = num
        break
    return index

# image1 = cv2.imread("./image-1.jpg")
# image2_ = cv2.imread("./image-2.jpg")

def change(image1, image2_):
    resize_max = np.max([image1.shape[1], image1.shape[0], image2_.shape[1], image2_.shape[0]]) 

    image1 = cv2.resize(image1, (resize_max,resize_max))
    img1 = np.array(image1)
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    mask = np.zeros_like(img1_gray)

    image2 = cv2.resize(image2_, (resize_max,resize_max))
    img2 = np.array(image2)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    img1_points = detector(image1, 1)
    img2_points = detector(image2, 2)

    if len(img1_points) == 0 or len(img2_points) == 0:
        return image2_

    height, width, channels = img2.shape
    img2_new_face = np.zeros((height, width, channels), np.uint8)

    convexhull = cv2.convexHull(img1_points)
    convexhull2 = cv2.convexHull(img2_points)
    cv2.fillConvexPoly(mask, convexhull, 255)  
    face_image_1 = cv2.bitwise_and(img1, img1, mask=mask)

    rect = cv2.boundingRect(convexhull)
    subdiv = cv2.Subdiv2D(rect)
    subdiv.insert(img1_points.tolist())
    triangles = subdiv.getTriangleList()
    triangles = np.array(triangles, dtype=np.int32)
        
    triangles_id = []
    for t in triangles:
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        index_pt1 = np.where((img1_points == pt1).all(axis=1))
        index_pt1 = extract_index_nparray(index_pt1)

        index_pt2 = np.where((img1_points == pt2).all(axis=1))
        index_pt2 = extract_index_nparray(index_pt2)

        index_pt3 = np.where((img1_points == pt3).all(axis=1))
        index_pt3 = extract_index_nparray(index_pt3)
            
        if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
            triangle = [index_pt1, index_pt2, index_pt3]
            triangles_id.append(triangle)

    img2_new_face = np.zeros_like(img2, np.uint8)
    for triangle_index in triangles_id:

        tr1_pt1 = img1_points[triangle_index[0]]
        tr1_pt2 = img1_points[triangle_index[1]]
        tr1_pt3 = img1_points[triangle_index[2]]
        triangle1 = np.array([tr1_pt1, tr1_pt2, tr1_pt3], np.int32)
        rect1 = cv2.boundingRect(triangle1)
        (x1, y1, w1, h1) = rect1
        cropped_triangle = img1[y1: y1 + h1, x1: x1 + w1]
        cropped_tr1_mask = np.zeros((h1, w1), np.uint8)
        points = np.array([[tr1_pt1[0] - x1, tr1_pt1[1] - y1],
                        [tr1_pt2[0] - x1, tr1_pt2[1] - y1],
                        [tr1_pt3[0] - x1, tr1_pt3[1] - y1]], np.int32)
        cv2.fillConvexPoly(cropped_tr1_mask, points, 255)
        cropped_triangle = cv2.bitwise_and(cropped_triangle, cropped_triangle,
                                        mask=cropped_tr1_mask)

        tr2_pt1 = img2_points[triangle_index[0]]
        tr2_pt2 = img2_points[triangle_index[1]]
        tr2_pt3 = img2_points[triangle_index[2]]
        triangle2 = np.array([tr2_pt1, tr2_pt2, tr2_pt3], np.int32)
        rect2 = cv2.boundingRect(triangle2)
        (x2, y2, w2, h2) = rect2
        cropped_triangle2 = img2[y2: y2 + h2, x2: x2 + w2]
        cropped_tr2_mask = np.zeros((h2, w2), np.uint8)
        points2 = np.array([[tr2_pt1[0] - x2, tr2_pt1[1] - y2],
                        [tr2_pt2[0] - x2, tr2_pt2[1] - y2],
                        [tr2_pt3[0] - x2, tr2_pt3[1] - y2]], np.int32)
        cv2.fillConvexPoly(cropped_tr2_mask, points2, 255)
        #print(img2.shape,[tr2_pt1, tr2_pt2, tr2_pt3], y2, y2 + h2, cropped_triangle2.shape, cropped_tr2_mask.shape)
        cropped_triangle2 = cv2.bitwise_and(cropped_triangle2, cropped_triangle2,
                                        mask=cropped_tr2_mask)

        points = np.float32(points)
        points2 = np.float32(points2)
        M = cv2.getAffineTransform(points, points2)
        warped_triangle = cv2.warpAffine(cropped_triangle, M, (w2, h2))
        warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=cropped_tr2_mask)


        img2_new_face_rect_area = img2_new_face[y2: y2 + h2, x2: x2 + w2]
        img2_new_face_rect_area_gray = cv2.cvtColor(img2_new_face_rect_area, cv2.COLOR_BGR2GRAY)
        
        _, mask_triangles_designed = cv2.threshold(img2_new_face_rect_area_gray, 1, 255, cv2.THRESH_BINARY_INV)
        warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=mask_triangles_designed)

        img2_new_face_rect_area = cv2.add(img2_new_face_rect_area, warped_triangle)
        img2_new_face[y2: y2 + h2, x2: x2 + w2] = img2_new_face_rect_area

    img2_face_mask = np.zeros_like(img2_gray)
    img2_head_mask = cv2.fillConvexPoly(img2_face_mask, convexhull2, 255)
    img2_face_mask = cv2.bitwise_not(img2_head_mask)
    img2_noface = cv2.bitwise_and(img2, img2, mask=img2_face_mask)
    img2_new_face_blur = cv2.medianBlur(img2_new_face, 3)

    result = cv2.add(img2_noface, img2_new_face_blur)

    (x3, y3, w3, h3) = cv2.boundingRect(convexhull2)
    center_face2 = (int((x3 + x3 + w3) / 2), int((y3 + y3 + h3) / 2))
    seamlessclone = cv2.seamlessClone(result, img2, img2_head_mask, center_face2, cv2.NORMAL_CLONE)
    #seamlessclone = result
    seamlessclone = cv2.resize(seamlessclone, (image2_.shape[1], image2_.shape[0]))

    return seamlessclone
