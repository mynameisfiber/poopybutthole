import numpy as np
import cv2


FACE_POINTS = list(range(17, 68))
MOUTH_POINTS = list(range(48, 61))
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
NOSE_POINTS = list(range(27, 35))
JAW_POINTS = list(range(0, 17))

# Points used to line up the images.
ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS +
                RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS)

# Points from the second image to overlay on the first. The convex hull of each
# element will be overlaid.
OVERLAY_POINTS = [
    LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS + RIGHT_BROW_POINTS,
    NOSE_POINTS + MOUTH_POINTS,
]

def faceswap(face_image, face_landmarks, target_image, target_face, target_landmarks,
             feather_amount=0.1, color_correct_blur_frac=0.8):
    """
    Perform a face swap where face_image and face_landmarks are the faces and
    face landmark points for the desired.  Target image and target landmarks
    point to the face to be changed. Feather amount is the percentage of the
    face to feather the face-mask for a nicer transition and
    color_correct_blur_frac is the amount of blur to use during color
    correction, as a fraction of the pupillary distance.
    """
    M = transformation_from_points(target_landmarks[ALIGN_POINTS],
                                   face_landmarks[ALIGN_POINTS])
    
    feather = int(max(target_face.width(), target_face.height()) *
            feather_amount)
    feather |= 0b1 # make odd

    mask = get_face_mask(face_image, face_landmarks, feather)
    warped_mask = warp_im(mask, M, target_image.shape)
    combined_mask = np.max([
        get_face_mask(target_image, target_landmarks, feather), 
        warped_mask
    ], axis=0)
    
    warped_target_image = warp_im(face_image, M, target_image.shape)

    warped_corrected_target_image = correct_colours(target_image,
            warped_target_image, target_landmarks, color_correct_blur_frac)
    
    output_im = target_image * (1.0 - combined_mask) + warped_corrected_target_image * combined_mask
    return output_im

def transformation_from_points(points1, points2):
    """
    Return an affine transformation [s * R | T] such that:

        sum ||s*R*p1,i + T - p2,i||^2

    is minimized.

    """
    # Solve the procrustes problem by subtracting centroids, scaling by the
    # standard deviation, and then using the SVD to calculate the rotation. See
    # the following for more details:
    #   https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem

    points1 = np.asmatrix(points1, np.float64)
    points2 = np.asmatrix(points2, np.float64)

    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2

    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2

    U, S, Vt = np.linalg.svd(points1.T * points2)

    # The R we seek is in fact the transpose of the one given by U * Vt. This
    # is because the above formulation assumes the matrix goes on the right
    # (with row vectors) where as our solution requires the matrix to be on the
    # left (with column vectors).
    R = (U * Vt).T

    return np.vstack([
        np.hstack(
            ((s2 / s1) * R,
            c2.T - (s2 / s1) * R * c1.T)
        ),
        np.array([0., 0., 1.])
    ])

def get_face_mask(im, landmarks, feather_amount):
    im = np.zeros(im.shape[:2], dtype=np.float64)

    for group in OVERLAY_POINTS:
        draw_convex_hull(im, landmarks[group], color=1)

    im = np.array([im, im, im]).transpose((1, 2, 0))

    im = (cv2.GaussianBlur(im, (feather_amount, feather_amount), 0) > 0) * 1.0
    im = cv2.GaussianBlur(im, (feather_amount, feather_amount), 0)

    return im
    
def warp_im(im, M, dshape):
    output_im = np.zeros(dshape, dtype=im.dtype)
    cv2.warpAffine(im,
                   M[:2],
                   (dshape[1], dshape[0]),
                   dst=output_im,
                   borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LANCZOS4)
    return output_im

def correct_colours(im1, im2, landmarks1, color_correct_blur_frac):
    blur_amount = color_correct_blur_frac * np.linalg.norm(
                              np.mean(landmarks1[LEFT_EYE_POINTS], axis=0) -
                              np.mean(landmarks1[RIGHT_EYE_POINTS], axis=0))
    blur_amount = int(blur_amount)
    if blur_amount % 2 == 0:
        blur_amount += 1
    im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
    im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)

    # Avoid divide-by-zero errors.
    im2_blur += 128 * (im2_blur <= 1.0)

    return (im2.astype(np.float64) * im1_blur.astype(np.float64) /
                                                im2_blur.astype(np.float64))

def draw_convex_hull(im, points, color):
    points = cv2.convexHull(points)
    cv2.fillConvexPoly(im, points, color=color)

