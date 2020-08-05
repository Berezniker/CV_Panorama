import numpy as np
from numpy.linalg import norm, inv, svd
from random import sample
from skimage.feature import ORB, match_descriptors
from skimage.color import rgb2gray
from skimage.transform import warp, ProjectiveTransform
from skimage.filters import gaussian
import matplotlib.pyplot as plt

DEFAULT_TRANSFORM = ProjectiveTransform


def show_image(image):
    # для отображения изображения в Pycharm
    plt.figure()
    plt.imshow(image)
    plt.show(block=True)


def find_orb(img, n_keypoints=256):
    """Find keypoints and their descriptors in image.

    img ((W, H, 3)  np.ndarray) : 3-channel image
    n_keypoints (int) : number of keypoints to find

    Returns:
        (N, 2)  np.ndarray : keypoints
        (N, 256)  np.ndarray, type=np.bool  : descriptors
    """

    descriptor_extractor = ORB(n_keypoints=n_keypoints)
    descriptor_extractor.detect_and_extract(rgb2gray(img))

    return descriptor_extractor.keypoints, descriptor_extractor.descriptors


def center_and_normalize_points(points):
    """Center the image points, such that the new coordinate system has its
    origin at the centroid of the image points.

    Normalize the image points, such that the mean distance from the points
    to the origin of the coordinate system is sqrt(2).

    points ((N, 2) np.ndarray) : the coordinates of the image points

    Returns:
        (3, 3) np.ndarray : the transformation matrix to obtain the new points
        (N, 2) np.ndarray : the transformed image points
    """

    pointsh = np.row_stack([points.T, np.ones((points.shape[0]), )])
    point_center = np.mean(points, axis=0)
    distance_mean = np.mean(norm(points - point_center, axis=1))
    normalization_coefficient = 2 ** 0.5 / distance_mean
    matrix = np.array([[normalization_coefficient, 0, -normalization_coefficient * point_center[0]],
                       [0, normalization_coefficient, -normalization_coefficient * point_center[1]],
                       [0, 0, 1]])

    transform_pointsh = np.dot(matrix, pointsh).transpose()
    transform_points = transform_pointsh[:, :2] / transform_pointsh[:, 2, np.newaxis]

    return matrix, transform_points


def find_homography(src_keypoints, dest_keypoints):
    """Estimate homography matrix from two sets of N (4+) corresponding points.

    src_keypoints ((N, 2) np.ndarray) : source coordinates
    dest_keypoints ((N, 2) np.ndarray) : destination coordinates

    Returns:
        ((3, 3) np.ndarray) : homography matrix
    """

    src_matrix, src = center_and_normalize_points(src_keypoints)
    dest_matrix, dest = center_and_normalize_points(dest_keypoints)

    A = np.empty((2 * src.shape[0], 9))
    A[::2, :2] = -src
    A[::2, 2] = -1
    A[::2, 3:6] = 0
    A[::2, 6:8] = src * dest[:, 0, np.newaxis]
    A[::2, 8] = dest[:, 0]
    A[1::2, :3] = 0
    A[1::2, 3:5] = -src
    A[1::2, 5] = -1
    A[1::2, 6:8] = src * dest[:, 1, np.newaxis]
    A[1::2, 8] = dest[:, 1]

    _, _, V = svd(A)
    H = V[-1].reshape(3, 3)
    homography_matrix = inv(dest_matrix).dot(H).dot(src_matrix)

    return homography_matrix


def ransac_transform(src_keypoints, src_descriptors, dest_keypoints, dest_descriptors,
                     max_trials=512, residual_threshold=3, return_matches=False):
    """Match keypoints of 2 images and find ProjectiveTransform using RANSAC algorithm.

    src_keypoints ((N, 2) np.ndarray) : source coordinates
    src_descriptors ((N, 256) np.ndarray) : source descriptors
    dest_keypoints ((N, 2) np.ndarray) : destination coordinates
    dest_descriptors ((N, 256) np.ndarray) : destination descriptors
    max_trials (int) : maximum number of iterations for random sample selection.
    residual_threshold (float) : maximum distance for a data point to be classified as an inlier.
    return_matches (bool) : if True function returns matches

    Returns:
        skimage.transform.ProjectiveTransform : transform of source image to destination image
        (Optional)(N, 2) np.ndarray : inliers' indexes of source and destination images
    """

    matches = match_descriptors(src_descriptors, dest_descriptors)
    homography_matrix, num_points = None, 4
    best_inliers, best_inlier_num, best_residual_sum = None, 0, np.inf

    for _ in range(max_trials):
        random_points_index = sample(range(matches.shape[0]), num_points)
        src_random_points = src_keypoints[matches[random_points_index, 0]]
        dest_random_points = dest_keypoints[matches[random_points_index, 1]]

        homography_matrix = find_homography(src_random_points, dest_random_points)

        residual = norm(dest_keypoints[matches[:, 1]] -
                        ProjectiveTransform(homography_matrix)(src_keypoints[matches[:, 0]]), axis=1)
        inliers = matches[residual < residual_threshold]
        inliers_num = inliers.shape[0]
        residual_sum = np.mean(residual)

        if inliers_num > best_inlier_num or \
                inliers_num == best_inlier_num and residual_sum < best_residual_sum:
            best_inliers = inliers
            best_inlier_num = inliers_num
            best_residual_sum = residual_sum

    src_inliers = src_keypoints[best_inliers[:, 0]]
    dst_inliers = dest_keypoints[best_inliers[:, 1]]
    homography_matrix = find_homography(src_inliers, dst_inliers)

    if return_matches:
        return ProjectiveTransform(homography_matrix), best_inliers
    else:
        return ProjectiveTransform(homography_matrix)


def find_simple_center_warps(forward_transforms):
    """Find transformations that transform each image to plane of the central image.

    forward_transforms (Tuple[N]) : - pairwise transformations

    Returns:
        Tuple[N + 1] : transformations to the plane of central image
    """

    image_count = len(forward_transforms) + 1
    center_index = (image_count - 1) // 2

    result = [None] * image_count
    result[center_index] = DEFAULT_TRANSFORM()
    for i in range(center_index - 1, -1, -1):
        result[i] = forward_transforms[i] + result[i + 1]
    for i in range(center_index + 1, image_count):
        forward_transforms[i - 1].params = inv(forward_transforms[i - 1].params)
        result[i] = forward_transforms[i - 1] + result[i - 1]

    return tuple(result)


def get_corners(image_collection, center_warps):
    """Get corners' coordinates after transformation."""

    for img, transform in zip(image_collection, center_warps):
        height, width, _ = img.shape
        corners = np.array([[0, 0],
                            [height, 0],
                            [height, width],
                            [0, width]])

        yield transform(corners)[:, ::-1]


def get_min_max_coords(corners):
    """Get minimum and maximum coordinates of corners."""
    corners = np.concatenate(corners)
    return corners.min(axis=0), corners.max(axis=0)


def get_final_center_warps(image_collection, simple_center_warps):
    """Find final transformations.

    image_collection (Tuple[N]) : list of all images
    simple_center_warps (Tuple[N])  : transformations unadjusted for shift

    Returns:
        Tuple[N] : final transformations
    """

    corners = tuple(get_corners(image_collection, simple_center_warps))
    min_coords, max_coords = get_min_max_coords(corners)
    output_shape = (max_coords[1] - min_coords[1], max_coords[0] - min_coords[0])
    shift_matrix = np.array([[1, 0, -1 * min_coords[1]],
                             [0, 1, -1 * min_coords[0]],
                             [0, 0, 1]])

    for transform in simple_center_warps:
        transform.params = shift_matrix.dot(transform.params)

    return simple_center_warps, output_shape


def rotate_transform_matrix(transform):
    """Rotate matrix so it can be applied to row:col coordinates."""
    matrix = transform.params[(1, 0, 2), :][:, (1, 0, 2)]
    return type(transform)(matrix)


def warp_image(image, transform, output_shape):
    """Apply transformation to an image and its mask

    image ((W, H, 3)  np.ndarray) : image for transformation
    transform (skimage.transform.ProjectiveTransform): transformation to apply
    output_shape (float, float) : shape of the final pano

    Returns:
        (W, H, 3) float64 np.ndarray : warped image
        (W, H) bool np.ndarray : warped mask
    """

    output_shape_int = (int(output_shape[0]), int(output_shape[1]))
    warped_image = warp(image, inv(rotate_transform_matrix(transform).params),
                        output_shape=output_shape_int, order=0)
    # order=0 : не будет швов при наложении в merge_pano
    warped_mask = rgb2gray(warped_image) != 0

    return warped_image, warped_mask


def merge_pano(image_collection, final_center_warps, output_shape):
    """ Merge the whole panorama

    image_collection (Tuple[N]) : list of all images
    final_center_warps (Tuple[N])  : transformations
    output_shape (float, float) : shape of the final pano

    Returns:
        (output_shape) np.ndarray: final pano
    """

    result = np.zeros((int(output_shape[0]), int(output_shape[1]), 3), dtype=np.float64)
    result_mask = np.zeros((int(output_shape[0]), int(output_shape[1])), dtype=np.bool8)

    for image, transform in zip(image_collection, final_center_warps):
        w_image, w_mask = warp_image(image, transform, output_shape)
        result_mask = result_mask ^ w_mask
        result += w_image * result_mask[:, :, np.newaxis]
        result_mask = result_mask | w_mask

    return (result * 255).astype(np.uint8)


def get_gaussian_pyramid(image, n_layers=4, sigma=1):
    """Get Gaussian pyramid.

    image ((W, H, 3)  np.ndarray) : original image
    n_layers (int) : number of layers in Gaussian pyramid
    sigma (int) : Gaussian sigma

    Returns:
        tuple(n_layers) Gaussian pyramid

    """

    gaussian_pyramid = [image]

    for layers in range(n_layers - 1):
        gaussian_pyramid.append(gaussian(gaussian_pyramid[-1], sigma))

    return gaussian_pyramid


def get_laplacian_pyramid(image, n_layers=4, sigma=1):
    """Get Laplacian pyramid

    image ((W, H, 3)  np.ndarray) : original image
    n_layers (int) : number of layers in Laplacian pyramid
    sigma (int) : Gaussian sigma

    Returns:
        tuple(n_layers) Laplacian pyramid
    """

    gaussian_pyramid = get_gaussian_pyramid(image, n_layers, sigma)
    laplacian_pyramid = []

    for i in range(n_layers - 1):
        laplacian_pyramid.append(gaussian_pyramid[i] - gaussian_pyramid[i + 1])
    laplacian_pyramid.append(gaussian_pyramid[-1])

    return laplacian_pyramid


def merge_laplacian_pyramid(laplacian_pyramid):
    """Recreate original image from Laplacian pyramid

    laplacian pyramid: tuple of np.array (h, w, 3)

    Returns:
        np.array (h, w, 3)
    """
    return sum(laplacian_pyramid)


def increase_contrast(image_collection):
    """Increase contrast of the images in collection"""

    result = []
    for img in image_collection:
        img = img.copy()
        for i in range(img.shape[-1]):
            img[:, :, i] -= img[:, :, i].min()
            img[:, :, i] /= img[:, :, i].max()
        result.append(img)

    return result


def get_3D_merge_mask(mask1, mask2):
    intersection_mask = (mask1 & mask2).astype(np.float64)
    l_index = np.where(np.sum(intersection_mask, axis=0) != 0)[0][0]
    r_index = np.where(np.sum(intersection_mask, axis=0) != 0)[0][-1]
    merge_mask = np.zeros_like(intersection_mask)
    merge_mask[:, (r_index + l_index) // 2:] = 1
    merge_mask = np.dstack((merge_mask, merge_mask, merge_mask)).astype(np.float64)

    return merge_mask


def gaussian_merge_pano(image_collection, final_center_warps, output_shape,
                        n_layers=4, image_sigma=3, merge_sigma=10):
    """ Merge the whole panorama using Laplacian pyramid

    image_collection (Tuple[N]) : list of all images
    final_center_warps (Tuple[N])  : transformations
    output_shape (float, float) : shape of the final pano
    n_layers (int) : number of layers in Laplacian pyramid
    image_sigma (int) :  sigma for Gaussian filter for images
    merge_sigma (int) : sigma for Gaussian filter for masks

    Returns:
        (output_shape) np.ndarray: final pano
    """

    contrast_image_collection = increase_contrast(image_collection)
    final_pano, result_mask = warp_image(contrast_image_collection[0], final_center_warps[0], output_shape)
    for image, transform in zip(contrast_image_collection[1:], final_center_warps[1:]):
        w_image, w_mask = warp_image(image, transform, output_shape)
        merge_mask = get_3D_merge_mask(result_mask, w_mask)

        pyr_gss_merge_mask = get_gaussian_pyramid(merge_mask, n_layers, merge_sigma)
        pyr_lap_pano = get_laplacian_pyramid(final_pano, n_layers, image_sigma)
        pyr_lap_w_image = get_laplacian_pyramid(w_image, n_layers, image_sigma)

        result_laplacian_pyramid = [img1 * (1 - mask) + img2 * mask for img1, img2, mask
                                    in zip(pyr_lap_pano, pyr_lap_w_image, pyr_gss_merge_mask)]
        final_pano = np.clip(merge_laplacian_pyramid(result_laplacian_pyramid), a_min=0, a_max=1)
        result_mask = result_mask | w_mask

    return (final_pano * 255).astype(np.uint8)


if __name__ == '__main__':
    from skimage import io

    pano_image_collection = io.ImageCollection('jpeg/lowres/8_*.jpg',
                                               load_func=lambda f: io.imread(f).astype(np.float64) / 255)
    keypoints, descriptors = zip(*(find_orb(img) for img in pano_image_collection))
    forward_transforms = tuple(ransac_transform(src_kp, src_desc, dest_kp, dest_desc)
                               for src_kp, src_desc, dest_kp, dest_desc
                               in zip(keypoints[:-1], descriptors[:-1], keypoints[1:], descriptors[1:]))

    simple_center_warps = find_simple_center_warps(forward_transforms)
    final_center_warps, output_shape = get_final_center_warps(pano_image_collection, simple_center_warps)
    # result = merge_pano(pano_image_collection, final_center_warps, output_shape)
    # show_image(result)
    result = gaussian_merge_pano(pano_image_collection, final_center_warps, output_shape)
    show_image(result)
