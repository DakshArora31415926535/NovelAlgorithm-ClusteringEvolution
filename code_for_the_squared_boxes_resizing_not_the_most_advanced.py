from PIL import Image, ImageDraw
import multiprocessing as mp
import math
import colorsys
import numpy as np
from skimage import measure

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

original_width = 100
original_height = 100

def rgb_to_hsv(r, g, b):
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    h = h * 360
    s = s * 100
    v = v * 100
    return h, s, v

def create_colored_squares_image(square_size=20):
    image = Image.new('RGB', (original_width, original_height))
    draw = ImageDraw.Draw(image)

    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    color_index = 0

    for y in range(0, original_height, square_size):
        for x in range(0, original_width, square_size):
            draw.rectangle([x, y, x + square_size - 1, y + square_size - 1], fill=colors[color_index % len(colors)])
            color_index += 1

    image_path = 'colored_squares_image.jpg'
    image.save(image_path)
    print(f"Created an image with colored squares with dimensions {original_width}x{original_height} and saved as {image_path}")

def is_similar_color(h1, s1, v1, h2, s2, v2, h_range=10, s_range=10, v_range=10):
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

def create_new_image(new_width, new_height, clusters, cluster_averages):
    new_image = Image.new('RGB', (new_width, new_height))
    total_pixels = new_width * new_height
    cluster_sizes = [len(cluster) for cluster in clusters]
    freex = []
    freey = []
    for i, cluster in enumerate(clusters):
        if freex != [] and freex[-1] >= new_width:
            freex = []
        if freey != [] and freey[-1] >= new_height:
            freey = []
        coords = [(x, y) for x, y, _ in cluster]

        x_min = min(coords[0] for coords in cluster)
        x_max = max(coords[0] for coords in cluster)
        y_min = min(coords[1] for coords in cluster)
        y_max = max(coords[1] for coords in cluster)

        width_ratio = (x_max - x_min + 1) / original_width
        height_ratio = (y_max - y_min + 1) / original_height

        new_x_min = int((x_min / original_width) * new_width)
        new_x_max = min(new_x_min + int(width_ratio * new_width), new_width)
        new_y_min = int((y_min / original_height) * new_height)
        new_y_max = min(new_y_min + int(height_ratio * new_height), new_height)
        if freex == []:
            freex.append(new_x_max)
        if freex != [] and new_x_max > freex[-1]:
            freex.append(new_x_max)
        if freey != [] and new_y_max > freey[-1]:
            freey.append(new_y_max)

        cluster_color = cluster_averages[i]
        h, s, v = cluster_color
        r, g, b = [int(val) for val in hsv_to_rgb(h, s, v)]

        for x in range(new_x_min, new_x_max):
            for y in range(new_y_min, new_y_max):
                if x < new_width and y < new_height:
                    new_image.putpixel((x, y), (r, g, b))

    new_image.save('new_image.jpg')
    print(f"Created a new image with dimensions {new_width}x{new_height} and saved as new_image.jpg")

def hsv_to_rgb(h, s, v):
    h /= 360.0
    s /= 100.0
    v /= 100.0
    i = int(h * 6.0)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - (s * f))
    t = v * (1.0 - (s * (1.0 - f)))
    i %= 6
    if i == 0:
        r, g, b = v, t, p
    elif i == 1:
        r, g, b = q, v, p
    elif i == 2:
        r, g, b = p, v, t
    elif i == 3:
        r, g, b = p, q, v
    elif i == 4:
        r, g, b = t, p, v
    elif i == 5:
        r, g, b = v, p, q
    return int(r * 255), int(g * 255), int(b * 255)

def main():
    create_colored_squares_image()
    image = Image.open("colored_squares_image.jpg")
    rgb_image = image.convert("RGB")
    width, height = rgb_image.size

    pool = mp.Pool(mp.cpu_count())

    hsv_values = pool.map(process_pixel, [(x, y, rgb_image) for x in range(width) for y in range(height)])

    hsv_array = [[None for _ in range(width)] for _ in range(height)]
    for x, y, hsv in hsv_values:
        hsv_array[y][x] = (x, y, hsv)

    pool.close()
    pool.join()

    clusters = []
    cluster_averages = []

    for y in range(0, original_height):
        for x in range(0, original_height):
            x_coord, y_coord, hsv = hsv_array[y][x]
            h, s, v = hsv
            found_cluster = False
            for i in range(len(cluster_averages)):
                cluster_avg = cluster_averages[i]
                if is_similar_color(h, s, v, *cluster_avg, h_range=10, s_range=10, v_range=10):
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

    new_width = 300
    new_height = 300
    create_new_image(new_width, new_height, clusters, cluster_averages)

if __name__ == "__main__":
    main()


