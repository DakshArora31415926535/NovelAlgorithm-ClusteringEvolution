# def calculate_intersections(cluster_points, boundary_points, d1, d2):
#     # Calculate the center of the cluster
#     center_x = sum(p[0] for p in cluster_points) / len(cluster_points)
#     center_y = sum(p[1] for p in cluster_points) / len(cluster_points)
#     center_z = sum(p[2][2] for p in cluster_points) / len(cluster_points)
#     center = (center_x, center_y, center_z)
#
#     intersections = []
#
#     for boundary_point in boundary_points:
#         x1, y1, z1 = center
#         x2, y2, z2 = boundary_point[:3]
#
#         # Calculate the direction vector of the line from the center to the boundary point
#         direction_vector = (x2 - x1, y2 - y1, z2 - z1)
#
#         # Calculate the parameter t for the intersection with the third plane
#         t = (d1 + d2 - x1) / direction_vector[0]
#
#         # Calculate the intersection point on the third plane
#         intersection_y = y1 + t * direction_vector[1]
#         intersection_z = z1 + t * direction_vector[2]
#
#         # Append the intersection point with the original color information from the boundary point
#         intersections.append((d1 + d2, intersection_y, intersection_z, boundary_point[2]))
#
#     return intersections
#
# # Example usage:
# cluster_points = [
#     (1, 0, (0, 0, 0)), (1, 1, (1, 1, 1)), (1, 2, (2, 2, 2)),
#     (1, 3, (3, 1, 1)), (1, 2, (2, 0, 0)), (1, -1, (1, -1, -1))
# ]
#
# boundary_points = [
#     (1, 3, (3, 1, 1)), (1, -1, (1, -1, -1))
# ]
#
# d1 = 1
# d2 = 2
#
# intersections = calculate_intersections(cluster_points, boundary_points, d1, d2)
# print(intersections)










# import matplotlib.pyplot as plt
#
# def calculate_intersections_with_global_center(cluster_points, boundary_points, d1, d2, cluster_average, center_x, center_y):
#     center_z = 0  # Light source is at z = 0
#     intersections = []
#
#     for boundary_point in boundary_points:
#         x_b, y_b, hsv = boundary_point
#         z_b = d1  # Boundary points are on the first plane at z = d1
#
#         # Direction vector from light source to boundary point
#         direction_vector = (x_b - center_x, y_b - center_y, z_b - center_z)
#
#         # Calculate the scaling factor t to reach the second plane at z = d1 + d2
#         t = (d1 + d2) / d1
#
#         # Calculate intersection coordinates
#         intersection_x = center_x + t * direction_vector[0]
#         intersection_y = center_y + t * direction_vector[1]
#         intersection_z = d1 + d2
#
#         # Append results with cluster average
#         intersections.append([intersection_x, intersection_y, intersection_z, cluster_average])
#
#     return intersections
#
# # Sample data
# x = []
# y = []
# x2 = []
# y2 = []
# d1 = 100
# d2 = 100
# cluster1 = [(0, 0, (0.0, 0.0, 12.156862745098039)), (0, 7, (0.0, 0.0, 12.549019607843137)), (1, 7, (0.0, 0.0, 12.549019607843137)), (9, 3, (0.0, 0.0, 12.549019607843137)), (11, 1, (0.0, 0.0, 12.549019607843137)), (11, 0, (0.0, 0.0, 12.549019607843137))]
# clusters = [(0, 0, (0.0, 0.0, 12.156862745098039)), (1, 0, (0.0, 0.0, 12.156862745098039)), (2, 0, (0.0, 0.0, 12.156862745098039)), (3, 0, (0.0, 0.0, 12.156862745098039)), (4, 0, (0.0, 0.0, 12.156862745098039)), (5, 0, (0.0, 0.0, 12.156862745098039)), (6, 0, (0.0, 0.0, 12.549019607843137)), (7, 0, (0.0, 0.0, 12.549019607843137)), (8, 0, (0.0, 0.0, 12.549019607843137)), (9, 0, (0.0, 0.0, 12.549019607843137)), (10, 0, (0.0, 0.0, 12.549019607843137)), (11, 0, (0.0, 0.0, 12.549019607843137)), (0, 1, (0.0, 0.0, 12.549019607843137)), (1, 1, (0.0, 0.0, 12.156862745098039)), (2, 1, (0.0, 0.0, 11.76470588235294)), (3, 1, (0.0, 0.0, 12.156862745098039)), (4, 1, (0.0, 0.0, 12.941176470588237)), (5, 1, (0.0, 0.0, 12.941176470588237)), (6, 1, (0.0, 0.0, 12.549019607843137)), (7, 1, (0.0, 0.0, 12.549019607843137)), (8, 1, (0.0, 0.0, 12.549019607843137)), (9, 1, (0.0, 0.0, 12.549019607843137)), (10, 1, (0.0, 0.0, 12.549019607843137)), (11, 1, (0.0, 0.0, 12.549019607843137)), (0, 2, (0.0, 0.0, 12.549019607843137)), (1, 2, (0.0, 0.0, 12.549019607843137)), (2, 2, (0.0, 0.0, 12.549019607843137)), (3, 2, (0.0, 0.0, 12.549019607843137)), (4, 2, (0.0, 0.0, 12.549019607843137)), (5, 2, (0.0, 0.0, 12.549019607843137)), (6, 2, (0.0, 0.0, 12.549019607843137)), (7, 2, (0.0, 0.0, 12.549019607843137)), (8, 2, (0.0, 0.0, 12.549019607843137)), (9, 2, (0.0, 0.0, 12.549019607843137)), (0, 3, (0.0, 0.0, 12.156862745098039)), (1, 3, (0.0, 0.0, 12.156862745098039)), (2, 3, (0.0, 0.0, 12.549019607843137)), (3, 3, (0.0, 0.0, 12.549019607843137)), (4, 3, (0.0, 0.0, 12.549019607843137)), (5, 3, (0.0, 0.0, 12.549019607843137)), (6, 3, (0.0, 0.0, 12.549019607843137)), (7, 3, (0.0, 0.0, 12.549019607843137)), (8, 3, (0.0, 0.0, 12.549019607843137)), (9, 3, (0.0, 0.0, 12.549019607843137)), (0, 4, (0.0, 0.0, 12.156862745098039)), (1, 4, (0.0, 0.0, 12.156862745098039)), (2, 4, (0.0, 0.0, 12.549019607843137)), (3, 4, (0.0, 0.0, 12.549019607843137)), (0, 5, (0.0, 0.0, 12.549019607843137)), (1, 5, (0.0, 0.0, 12.549019607843137)), (2, 5, (0.0, 0.0, 12.549019607843137)), (3, 5, (0.0, 0.0, 12.549019607843137)), (0, 6, (0.0, 0.0, 12.549019607843137)), (1, 6, (0.0, 0.0, 12.549019607843137)), (0, 7, (0.0, 0.0, 12.549019607843137)), (1, 7, (0.0, 0.0, 12.549019607843137))]
# cluster_averages = [(0.0, 0.0, 12.156862745098039)]
#
# boundary_points = cluster1
# intersections = calculate_intersections_with_global_center(clusters, boundary_points, d1, d2,
#                                                                cluster_averages[0], 125, 83)
# for i in range(0,len(intersections)):
#     print(intersections[i],end=", ")
# print("\n\n\n\n\n")
# mainx=intersections[0][0]
# mainy=intersections[0][1]
# for i in range(0,len(intersections)):
#     intersections[i][0]=intersections[i][0]+((-1)*mainx)
#     intersections[i][1] = intersections[i][1] + ((-1) * mainy)
# for i in range(0,len(intersections)):
#     print(intersections[i],end=", ")
# # Prepare data for plotting
# # for i in range(len(list1)):
# #     x.append(list1[i][0])
# #     y.append(167 - list1[i][1])
#
# for i in range(len(cluster1)):
#     x.append(cluster1[i][0])
#     y.append(167 - cluster1[i][1])
#
# for i in range(len(intersections)):
#     x2.append(intersections[i][0])
#     y2.append(334 - intersections[i][1])
#
# # Create subplots
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
#
# # Plot on the first subplot
# ax1.plot(x, y, 'bo-')  # 'bo-' means blue color, circle markers, and solid line
# ax1.set_xlim(0, 250)  # Set x-axis limits from 0 to 250
# ax1.set_ylim(0, 167)  # Set y-axis limits from 0 to 167
# ax1.set_xlabel('X axis label')
# ax1.set_ylabel('Y axis label')
# ax1.set_title('Plot of cluster1 ')
#
# # Plot on the second subplot
# ax2.plot(x2, y2, 'ro-')  # 'ro-' means red color, circle markers, and solid line
# ax2.set_xlim(0, 500)  # Set x-axis limits from 0 to 250
# ax2.set_ylim(0, 335)  # Set y-axis limits from 0 to 167
# ax2.set_xlabel('X axis label')
# ax2.set_ylabel('Y axis label')
# ax2.set_title('Plot of intersection of cluster 1')
#
# # Display the plot
# plt.tight_layout()
# plt.show()







import matplotlib.pyplot as plt
import numpy as np

def calculate_intersections_with_global_center(cluster_points, boundary_points, d1, d2, cluster_average, center_x, center_y):
    center_z = 0  # Light source is at z = 0
    intersections = []

    for boundary_point in boundary_points:
        x_b, y_b, hsv = boundary_point
        x_b,y_b=x_b-125,y_b-83
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
    mainx = intersections[0][0]
    mainy = intersections[0][1]
    print(mainx, "njsbhjcbdbh")
    print(mainy, "jdbhjjchjcbhjbhjvbjh")
    for i in range(len(intersections)):
        intersections[i][0] -= mainx
        intersections[i][1] -= mainy
    return intersections



current_width=250
current_height=167
desired_height=335
# Sample data
x = []
y = []
x2 = []
y2 = []
d1 = 100
d2 = 100
cluster1 = [(0, 0, (0.0, 0.0, 12.156862745098039)), (0, 7, (0.0, 0.0, 12.549019607843137)),
            (1, 7, (0.0, 0.0, 12.549019607843137)), (9, 3, (0.0, 0.0, 12.549019607843137)),
            (11, 1, (0.0, 0.0, 12.549019607843137)), (11, 0, (0.0, 0.0, 12.549019607843137))]
clusters = []
maincluster=[(0, 8, (149.99999999999994, 6.06060606060607, 12.941176470588237)),(0, 19, (150.0000000000001, 5.555555555555555, 14.117647058823529)),(104, 22, (150.0000000000001, 4.3478260869565215, 18.03921568627451)),(111, 22, (149.9999999999999, 4.25531914893617, 18.43137254901961)),(167, 7, (150.0000000000001, 5.0, 15.686274509803921)),(168, 4, (150.0000000000001, 5.555555555555555, 14.117647058823529)),(127, 0, (149.9999999999999, 5.7142857142857135, 13.725490196078432)),(12, 0, (150.0000000000001, 5.88235294117647, 13.333333333333334)),(4, 4, (149.99999999999994, 6.06060606060607, 12.941176470588237))]
cluster_averages = [(149.99999999999994, 6.06060606060607, 12.941176470588237)]

boundary_points = maincluster
intersections = calculate_intersections_with_global_center(clusters, boundary_points, d1, d2,
                                                           cluster_averages[0], current_width/2, current_height/2)



print("\n\n\n\n before processong \n\n\n\n",intersections)
# Adjusting intersection points by shifting them so that the first point is at the origin
# mainx = intersections[0][0]
# mainy = intersections[0][1]
# print(mainx,"njsbhjcbdbh")
# print(mainy,"jdbhjjchjcbhjbhjvbjh")
# for i in range(len(intersections)):
#     intersections[i][0] -= mainx
#     intersections[i][1] -= mainy

# print("\n\n\n\n after processong \n\n\n\n",intersections)

# Prepare data for plotting
# for i in range(len(cluster1)):
#     x.append(cluster1[i][0])
#     y.append(167 - cluster1[i][1])


for i in range(len(maincluster)):
    x.append(maincluster[i][0])
    y.append(current_height - maincluster[i][1])

for i in range(len(intersections)):
    x2.append(intersections[i][0])
    y2.append(desired_height - intersections[i][1])

# Create subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

# Plot on the first subplot
ax1.plot(x, y, 'bo-')  # 'bo-' means blue color, circle markers, and solid line
ax1.set_xlim(0, 250)  # Set x-axis limits from 0 to 250
ax1.set_ylim(0, 167)  # Set y-axis limits from 0 to 167
ax1.set_xlabel('X axis label')
ax1.set_ylabel('Y axis label')
ax1.set_title('Plot of cluster1')

# Plot on the second subplot
ax2.plot(x2, y2, 'ro-')  # 'ro-' means red color, circle markers, and solid line
ax2.set_xlim(0, 500)  # Set x-axis limits from 0 to 500
ax2.set_ylim(0, 350)  # Set y-axis limits from 0 to 335
ax2.set_xlabel('X axis label')
ax2.set_ylabel('Y axis label')
ax2.set_title('Plot of intersection of cluster 1')

# Display the plot
plt.tight_layout()
plt.show()
