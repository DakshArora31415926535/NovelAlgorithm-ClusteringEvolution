from PIL import Image, ImageDraw
import multiprocessing as mp
import numpy as np
from skimage import measure
import colorsys

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

def calculate_intersections(cluster_points, boundary_points, d1, d2, cluster_average):
    center_x = sum(p[0] for p in cluster_points) / len(cluster_points)
    center_y = sum(p[1] for p in cluster_points) / len(cluster_points)
    center_z = d1  # All points in the cluster have the same z-value, which is d1
    center = (center_x, center_y, center_z)

    intersections = []

    for boundary_point in boundary_points:
        x_b, y_b, hsv = boundary_point
        z_b = d1

        direction_vector = (x_b - center_x, y_b - center_y, z_b - center_z)

        t = (d1 + d2) / d1

        intersection_x = center_x + t * direction_vector[0]
        intersection_y = center_y + t * direction_vector[1]
        intersection_z = d1 + d2

        intersections.append((intersection_x, intersection_y, intersection_z, cluster_average))

    return intersections

def fill_cluster_area(draw, intersections, cluster_average):
    if len(intersections) < 2:
        return  # Skip drawing if there are less than 2 points

    points = [(x, y) for x, y, z, avg in intersections]
    avg_h, avg_s, avg_v = cluster_average
    avg_rgb = tuple(int(c * 255) for c in colorsys.hsv_to_rgb(avg_h / 360, avg_s / 100, avg_v / 100))

    draw.polygon(points, fill=avg_rgb)

def main():
    image_path = "images/input_image_to_be_analysed.jpg"
    desired_width = 500  # Example desired width
    d1 = 50  # Predefined distance between the first and second planes

    clusters, cluster_averages, current_width, current_height = load_and_cluster_image(image_path)

    d2, calculated_width, calculated_height = calculate_distance_between_planes(image_path, desired_width, d1)
    print(f"The calculated distance between the second and third planes (d2) is: {d2}")
    print(f"The calculated width is: {calculated_width}, and the calculated height is: {calculated_height}")

    new_image = Image.new("RGB", (int(calculated_width), int(calculated_height)))
    draw = ImageDraw.Draw(new_image)

    for i in range(len(clusters)):
        boundary_points = gift_wrapping(clusters[i])
        print(f"Cluster {i + 1} Boundary Points: {boundary_points}")

        intersections = calculate_intersections(clusters[i], boundary_points, d1, d2, cluster_averages[i])
        print(f"Cluster {i + 1} Intersections: {intersections}")

        fill_cluster_area(draw, intersections, cluster_averages[i])

    new_image.save("new_image.jpg")  # Save the new image

if __name__ == "__main__":
    main()