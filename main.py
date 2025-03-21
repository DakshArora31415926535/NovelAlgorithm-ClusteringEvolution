from PIL import Image, ImageDraw
import multiprocessing as mp
import numpy as np
from skimage import measure
import colorsys
from scipy.spatial import distance
from scipy.spatial.distance import euclidean
import numpy as np
from PIL import Image

def add_random_noise(image, noise_level=20):
    """
    Add random noise to an image.

    Parameters:
    - image: PIL.Image object, the input image.
    - noise_level: int, the intensity of the noise.

    Returns:
    - PIL.Image object, the image with added noise.
    """
    # Convert the image to a NumPy array
    img_array = np.array(image)

    # Generate random noise
    noise = np.random.randint(-noise_level, noise_level, img_array.shape, dtype='int16')

    # Add the noise to the image
    noisy_img_array = img_array.astype('int16') + noise

    # Clip the values to be in the valid range [0, 255] and convert back to uint8
    noisy_img_array = np.clip(noisy_img_array, 0, 255).astype('uint8')

    # Convert the array back to an image
    noisy_image = Image.fromarray(noisy_img_array)

    return noisy_image

def fill_empty_pixels_x_axis(image):
    width, height = image.size
    image_data = image.load()

    for y in range(height):
        last_color = (0, 0, 0)  # Start with white as the initial "last color"
        for x in range(width):
            current_color = image_data[x, y]
            if current_color == (0, 0, 0):  # Check if the pixel is white (empty)
                image_data[x, y] = last_color
            else:
                last_color = current_color

    return image

# Example usage:
# new_image = Image.open("new_image_generated_by_code_with_global_center_filled.jpg")
# filled_image = fill_empty_pixels_x_axis(new_image)
# filled_image.save("new_image_filled_x_axis.jpg")
# print("Empty pixels filled along x-axis and saved as 'new_image_filled_x_axis.jpg'")




# def fill_empty_spaces_with_nearest_cluster(new_image, clusters, cluster_averages, image_size):
#     width, height = image_size
#     new_image_data = new_image.load()
#
#     # Convert cluster data to a lookup for quick access
#     cluster_map = {}
#     for i, cluster in enumerate(clusters):
#         avg_rgb = tuple(int(c * 255) for c in
#                         colorsys.hsv_to_rgb(cluster_averages[i][0] / 360, cluster_averages[i][1] / 100,
#                                             cluster_averages[i][2] / 100))
#         for x, y, _ in cluster:
#             cluster_map[(x, y)] = avg_rgb
#
#     # Check all pixels in the new image
#     for x in range(width):
#         for y in range(height):
#             # If the pixel is not part of a cluster (empty), fill it
#             if (x, y) not in cluster_map:
#                 nearest_pixel = find_nearest_non_empty_pixel((x, y), cluster_map)
#                 if nearest_pixel:
#                     new_image_data[x, y] = cluster_map[nearest_pixel]
#
# def find_nearest_non_empty_pixel(empty_pixel, cluster_map):
#     empty_x, empty_y = empty_pixel
#     nearest_pixel = None
#     min_distance = float('inf')
#
#     for (x, y) in cluster_map.keys():
#         if cluster_map[(x, y)] != (255, 255, 255):  # Ensure that the pixel is non-empty (not white)
#             dist = euclidean((empty_x, empty_y), (x, y))
#             if dist < min_distance:
#                 min_distance = dist
#                 nearest_pixel = (x, y)
#
#     return nearest_pixel

# Helper functions
def rgb_to_hsv(r, g, b):
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    h = h * 360
    s = s * 100
    v = v * 100
    return h, s, v

def is_similar_color(h1, s1, v1, h2, s2, v2, h_range=5, s_range=5, v_range=5):
    return abs(h1 - h2) <= h_range and abs(s1 - s2) <= s_range and abs(v1 - v2) <= v_range

def update_cluster_average(cluster):
    num_colors = len(cluster)
    total_h = total_s = total_v = 0
    for i in range(num_colors):
        total_h += cluster[i][2][0]
        total_s += cluster[i][2][1]
        total_v += cluster[i][2][2]
    avg_h = total_h / num_colors
    avg_s = total_s / num_colors
    avg_v = total_v / num_colors
    return avg_h, avg_s, avg_v

def process_pixel(args):
    x, y, rgb_image = args
    r, g, b = rgb_image.getpixel((x, y))
    return x, y, rgb_to_hsv(r, g, b)

def split_connected_components(cluster):
    xs, ys = zip(*[(x, y) for x, y, hsv in cluster])
    width, height = max(xs) + 1, max(ys) + 1

    binary_mask = np.zeros((height, width), dtype=np.uint8)
    for x, y, hsv in cluster:
        binary_mask[y, x] = 1

    labels = measure.label(binary_mask, connectivity=2)

    clusters_dict = {}
    for (x, y, hsv), label in zip(cluster, labels[binary_mask == 1]):
        if label not in clusters_dict:
            clusters_dict[label] = []
        clusters_dict[label].append((x, y, hsv))

    clusters_list = list(clusters_dict.values())

    return clusters_list

def load_and_cluster_image(image_path):
    image = Image.open(image_path)
    rgb_image = image.convert("RGB")
    width, height = rgb_image.size

    pool = mp.Pool(mp.cpu_count())
    hsv_values = pool.map(process_pixel, [(x, y, rgb_image) for x in range(width) for y in range(height)])
    pool.close()
    pool.join()

    hsv_array = [[None for _ in range(width)] for _ in range(height)]
    for x, y, hsv in hsv_values:
        hsv_array[y][x] = (x, y, hsv)

    clusters = []
    cluster_averages = []

    for y in range(height):
        for x in range(width):
            x_coord, y_coord, hsv = hsv_array[y][x]
            h, s, v = hsv
            found_cluster = False
            for i in range(len(cluster_averages)):
                cluster_avg = cluster_averages[i]
                if is_similar_color(h, s, v, *cluster_avg, h_range=5, s_range=5, v_range=5):
                    clusters[i].append((x_coord, y_coord, hsv))
                    cluster_averages[i] = update_cluster_average(clusters[i])
                    found_cluster = True
                    break
            if not found_cluster:
                clusters.append([(x_coord, y_coord, hsv)])
                cluster_averages.append((h, s, v))

    for i in range(len(clusters)):
        connected_components = split_connected_components(clusters[i])
        if len(connected_components) > 1:
            clusters[i] = connected_components[0]
            for j in range(1, len(connected_components)):
                clusters.append(connected_components[j])
                cluster_averages.append(update_cluster_average(connected_components[j]))

    return clusters, cluster_averages, width, height

def calculate_distance_between_planes(image_path, desired_width, d1):
    image = Image.open(image_path)
    current_width, current_height = image.size

    if current_width == 0 or current_height == 0:
        raise ValueError("Current width or height cannot be zero")

    aspect_ratio = current_width / current_height
    desired_height = desired_width / aspect_ratio
    scale_factor = desired_width / current_width

    d2 = scale_factor * d1 - d1
    return d2, desired_width, desired_height

def gift_wrapping(points):
    if len(points) < 3:
        return points

    if all(p[0] == points[0][0] for p in points):
        min_y_point = min(points, key=lambda p: p[1])
        max_y_point = max(points, key=lambda p: p[1])
        return [min_y_point, max_y_point]

    leftmost = min(points, key=lambda p: p[0])

    hull = []
    current = leftmost

    while True:
        hull.append(current)
        endpoint = points[0] if points[0] != current else points[1]

        for point in points:
            if endpoint == current or is_counter_clockwise(current, endpoint, point):
                endpoint = point
            elif is_collinear(current, endpoint, point):
                if distance(current, point) > distance(current, endpoint):
                    endpoint = point

        current = endpoint

        if current == leftmost:
            break

    return hull

def is_counter_clockwise(p1, p2, p3):
    return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]) > 0

def is_collinear(p1, p2, p3):
    return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]) == 0

def distance(p1, p2):
    return (p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2

def calculate_intersections_with_global_center(cluster_points, boundary_points, d1, d2, cluster_average, center_x, center_y,mainx,mainy):
    center_z = 0  # Light source is at z = 0
    intersections = []

    for boundary_point in boundary_points:
        x_b, y_b, hsv = boundary_point
        x_b, y_b = x_b - (center_x * 2) / 2, y_b - (center_y * 2) / 2
        z_b = d1  # Boundary points are on the first plane at z = d1

        # Direction vector from light source to boundary point
        direction_vector = np.array([x_b - center_x, y_b - center_y, z_b - center_z])

        # Calculate the scaling factor t to reach the second plane at z = d1 + d2
        t = (d1 + d2 - center_z) / (z_b - center_z)

        # Calculate intersection coordinates
        intersection_x = center_x + t * direction_vector[0]
        intersection_y = center_y + t * direction_vector[1]
        intersection_z = d1 + d2

        # Append results with cluster average
        intersections.append([intersection_x, intersection_y, intersection_z, cluster_average])

    for i in range(len(intersections)):
        intersections[i][0] -= mainx
        intersections[i][1] -= mainy

    return intersections

def calculate_intersections_with_global_center1(cluster_points, boundary_points, d1, d2, cluster_average, center_x, center_y):
    center_z = 0  # Light source is at z = 0
    intersections = []

    for boundary_point in boundary_points:
        x_b, y_b, hsv = boundary_point
        x_b, y_b = x_b - (center_x * 2) / 2, y_b - (center_y * 2) / 2
        z_b = d1  # Boundary points are on the first plane at z = d1

        # Direction vector from light source to boundary point
        direction_vector = np.array([x_b - center_x, y_b - center_y, z_b - center_z])

        # Calculate the scaling factor t to reach the second plane at z = d1 + d2
        t = (d1 + d2 - center_z) / (z_b - center_z)

        # Calculate intersection coordinates
        intersection_x = center_x + t * direction_vector[0]
        intersection_y = center_y + t * direction_vector[1]
        intersection_z = d1 + d2

        # Append results with cluster average
        intersections.append([intersection_x, intersection_y, intersection_z, cluster_average])

    return intersections

def fill_cluster_area(draw, intersections, cluster_average):
    if len(intersections) < 2:
        return  # Skip drawing if there are less than 2 points

    points = [(x, y) for x, y, z, avg in intersections]
    avg_h, avg_s, avg_v = cluster_average
    avg_rgb = tuple(int(c * 255) for c in colorsys.hsv_to_rgb(avg_h / 360, avg_s / 100, avg_v / 100))

    draw.polygon(points, fill=avg_rgb, outline=avg_rgb)  # Add outline for better visual distinction

# Main function
# def main():
#     image_path = "input_image_to_be_analysed.jpg"
#     desired_width = 500  # Example desired width
#     d1 = 100  # Predefined distance between the first and second planes
#
#     clusters, cluster_averages, current_width, current_height = load_and_cluster_image(image_path)
#
#     d2, calculated_width, calculated_height = calculate_distance_between_planes(image_path, desired_width, d1)
#     print(f"The calculated distance between the second and third planes (d2) is: {d2}")
#     print(f"The calculated width is: {calculated_width}, and the calculated height is: {calculated_height}")
#
#     new_image = Image.new("RGB", (int(calculated_width), int(calculated_height)))
#     draw = ImageDraw.Draw(new_image)
#
#     center_x = current_width / 2
#     center_y = current_height / 2
#
#     finalmainx = 0
#     finalmainy = 0
#
#     for i in range(len(clusters)):
#         boundary_points = gift_wrapping(clusters[i])
#         print(f"Cluster {i + 1} Boundary Points: {boundary_points}")
#
#         if i == 0:
#             intersections = calculate_intersections_with_global_center1(clusters[i], boundary_points, d1, d2,
#                                                                        cluster_averages[i], center_x, center_y)
#             finalmainx = intersections[0][0]
#             finalmainy = intersections[0][1]
#             for i in range(len(intersections)):
#                 intersections[i][0] -= finalmainx
#                 intersections[i][1] -= finalmainy
#         else:
#             intersections = calculate_intersections_with_global_center(clusters[i], boundary_points, d1, d2,
#                                                                        cluster_averages[i], center_x, center_y,
#                                                                        finalmainx, finalmainy)
#         print(f"Cluster {i + 1} Intersections: {intersections}")
#
#         fill_cluster_area(draw, intersections, cluster_averages[i])
#
#     new_image.save("new_image_with_global_center.jpg")
#     print("New image saved as 'new_image_with_global_center.jpg'")

def main():
    image_path = "images/input_image_to_be_analysed.jpg"
    desired_width = 500  # Example desired width
    d1 = 100  # Predefined distance between the first and second planes

    # Load and cluster the original image
    clusters, cluster_averages, current_width, current_height = load_and_cluster_image(image_path)

    # Calculate the distance and dimensions for the new image
    d2, calculated_width, calculated_height = calculate_distance_between_planes(image_path, desired_width, d1)
    print(f"The calculated distance between the second and third planes (d2) is: {d2}")
    print(f"The calculated width is: {calculated_width}, and the calculated height is: {calculated_height}")

    # Create a new image with the desired dimensions
    new_image = Image.new("RGB", (int(calculated_width), int(calculated_height)))
    draw = ImageDraw.Draw(new_image)

    # Calculate the center of the original image
    center_x = current_width / 2
    center_y = current_height / 2

    # Variables to hold the offset for positioning the clusters
    finalmainx = 0
    finalmainy = 0

    # Process each cluster
    for i in range(len(clusters)):
        boundary_points = gift_wrapping(clusters[i])
        print(f"Cluster {i + 1} Boundary Points: {boundary_points}")

        if i == 0:
            intersections = calculate_intersections_with_global_center1(clusters[i], boundary_points, d1, d2,
                                                                       cluster_averages[i], center_x, center_y)
            finalmainx = intersections[0][0]
            finalmainy = intersections[0][1]
            for j in range(len(intersections)):
                intersections[j][0] -= finalmainx
                intersections[j][1] -= finalmainy
        else:
            intersections = calculate_intersections_with_global_center(clusters[i], boundary_points, d1, d2,
                                                                       cluster_averages[i], center_x, center_y,
                                                                       finalmainx, finalmainy)
        print(f"Cluster {i + 1} Intersections: {intersections}")

        fill_cluster_area(draw, intersections, cluster_averages[i])
    new_image.save("generated_resized_image_using_dynamic_clustering_without_noise_filled.jpg")

    # Fill empty spaces with the nearest cluster's average color
    # fill_empty_spaces_with_nearest_cluster(new_image, clusters, cluster_averages, (int(calculated_width), int(calculated_height)))
    new_image=fill_empty_pixels_x_axis(new_image)
    new_image=add_random_noise(new_image,10)
    # Save the new image
    new_image.save("generated_resized_image_using_dynamic_clustering_noise_filled.jpg")
    print("New image saved as 'generated_resized_image_using_dynamic_clustering_noise_filled.jpg'")

if __name__ == "__main__":
    main()


