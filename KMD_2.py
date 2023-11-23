import pandas as pd
import matplotlib
import easygui
import mplcursors
import math
from math import sqrt
from matplotlib.widgets import CheckButtons
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import numpy as np
from matplotlib.patches import Polygon
from matplotlib.widgets import TextBox, Button
import time
import csv
import tkinter as tk

# drag and drop
def select_file(title):
    file_path = easygui.fileopenbox(msg=title, filetypes=["*.xls", "*.xlsx"])
    return file_path

# when a dot on graph is picked
def on_pick(event):
    ind = event.ind[0]
    data = event.artist.get_gid()[ind]

def convert_to_km_kmd(df, column_name):
    df[column_name] = pd.to_numeric(df[column_name], errors='coerce')
    df = df.dropna(subset=[column_name])  # ignore letters
    konstant = 0.99965856
    km_values = df[column_name] * konstant
    kmd_values = -(km_values - round(km_values, 0))*1000
    return km_values, kmd_values

# for searching scatters
def convert_single_mass_to_km_kmd(mass_value):
    konstant = 0.99965856
    km = mass_value * konstant
    kmd = -(km - round(km, 0))
    return km, kmd

def parallel_line(x1, y1, x2, y2, d):
    # calculate direction vector
    dx = x2 - x1
    dy = y2 - y1

    # normalize
    len_v = math.sqrt(dx ** 2 + dy ** 2)
    dx /= len_v
    dy /= len_v

    # new points
    nx1 = x1 + d * dy
    ny1 = y1 - d * dx
    nx2 = x2 + d * dy
    ny2 = y2 - d * dx

    return nx1, ny1, nx2, ny2

def point_to_line_distance(x1, y1, x2, y2, px, py):
    distance = (abs((y2 - y1) * px - (x2 - x1) * py + x2 * y1 - y2 * x1)) / (sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2))

    min_x, max_x = min(x1, x2), max(x1, x2)
    min_y, max_y = min(y1, y2), max(y1, y2)

    if min_x <= px <= max_x and min_y <= py <= max_y:
        return distance

def point_to_line_distance2(x1, y1, x2, y2, px, py):
    distance = (abs((y2 - y1) * px - (x2 - x1) * py + x2 * y1 - y2 * x1)) / (sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2))
    return distance

def perpendicular_lines(x1, y1, x2, y2, perp_length, scatter_points):
    points_within = {}
    perp_lines = []
    touched_scatter = {}  # dictionary to store which scatter point is touched by which perp line
    touched_count = 0  # counter for touched scatter points

    original_length = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    num_perp_lines = int(original_length)* 3
    max_distance = (original_length / num_perp_lines) / 2

    for i in range(num_perp_lines + 1):
        t = i / (num_perp_lines)
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)

        dx = - (y2 - y1) * perp_length / original_length
        dy = (x2 - x1) * perp_length / original_length

        x_perp1, y_perp1 = x + dx, y + dy
        x_perp2, y_perp2 = x - dx, y - dy

        for px, py in scatter_points:
            distance = point_to_line_distance(x_perp1, y_perp1, x_perp2, y_perp2, px, py)
            if distance is not None and distance <= max_distance:
                points_within[(px, py)] = distance
                if (px, py) not in touched_scatter:
                    touched_scatter[(px, py)] = i
                    touched_count += 1

    return perp_lines, points_within

def find_max_dist(df_all, KM_col, KMD_col):
    max_KM_ion = df_all[KM_col].max()
    min_KM_ion = df_all[KM_col].min()
    min_KMD_ion = df_all.loc[df_all[KM_col] == min_KM_ion, KMD_col].values[0]
    max_KMD_ion = df_all.loc[df_all[KM_col] == max_KM_ion, KMD_col].values[0]

    distances = abs((max_KM_ion - min_KM_ion) * (min_KMD_ion - df_all[KMD_col]) - (min_KM_ion - df_all[KM_col]) * (
            max_KMD_ion - min_KMD_ion)) / (math.sqrt((max_KM_ion - min_KM_ion) ** 2 + (max_KMD_ion - min_KMD_ion) ** 2))

    max_distance_index = distances.idxmax()
    max_distance = distances[max_distance_index]
    max_distance_point = (df_all.loc[max_distance_index, KM_col], df_all.loc[max_distance_index, KMD_col])

    return max_distance_point, max_distance

def append_to_csv(filename, points_b, distances_b, points_y, distances_y, points_b2, distances_b2, points_y2, distances_y2):
    points_b = list(points_b)
    points_y = list(points_y)
    points_b2 = list(points_b2)
    points_y2 = list(points_y2)

    max_length = max(len(points_b), len(distances_b), len(points_y), len(distances_y), len(points_b2),
                     len(distances_b2), len(points_y2), len(distances_y2))

    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Limit 1 b KM', 'Limit 1 b KMD', 'Limit 1 b distances', '', 'Limit 1 y KM', 'Limit 1 y KMD',
                         'Limit 1 y distances', '', 'Limit 2 b KM', 'Limit 2 b KMD', 'Limit 2 b distances', '',
                         'Limit 2 y KM', 'Limit 2 y KMD', 'Limit 2 y distances'])

        for i in range(max_length):
            row = []

            for points, distances in [(points_b, distances_b), (points_y, distances_y), (points_b2, distances_b2),
                                      (points_y2, distances_y2)]:
                point_x = "{:.4f}".format(points[i][0]) if i < len(points) and points[i][
                    0] is not None and not np.ma.is_masked(points[i][0]) and not math.isnan(points[i][0]) else ''
                point_y = "{:.4f}".format(points[i][1] / 1000) if i < len(points) and points[i][
                    1] is not None and not np.ma.is_masked(points[i][1]) and not math.isnan(points[i][1]) else ''
                distance = "{:.4f}".format(distances[i] / 1000) if i < len(distances) and distances[
                    i] is not None and not math.isnan(distances[i]) else ''

                row.extend([point_x, point_y, distance, ''])

            writer.writerow(row)

def plot_graph(file_all):
    # reading from datafile
    df_all = pd.read_excel(file_all, header=None, skiprows=1)
    df_all.columns = ['Mass', 'Intensity', 'Mass_b', 'Mass_y']

    # mass to KM and KMD
    df_all['KM'], df_all['KMD'] = convert_to_km_kmd(df_all, 'Mass')
    df_all['KM_b'], df_all['KMD_b'] = convert_to_km_kmd(df_all, 'Mass_b')
    df_all['KM_y'], df_all['KMD_y'] = convert_to_km_kmd(df_all, 'Mass_y')
    fig, ax = plt.subplots(figsize=(9, 6))

    # plot data
    all_points = ax.scatter(df_all['KM'], df_all['KMD'], color='lightblue', s=1)
    ax.scatter(df_all['KM_b'], df_all['KMD_b'], color='orange', s=6)
    ax.scatter(df_all['KM_y'], df_all['KMD_y'], color='green', s=6)

    # For picker functionality
    all_points.set_picker(True)
    all_points.set_pickradius(5)
    all_points.set_gid(df_all[['Mass', 'Intensity', 'KM', 'KMD']].values)
    fig.canvas.mpl_connect('pick_event', on_pick)

    # For your perpendicular line calculations
    scatters = list(zip(df_all['KM'].values, df_all['KMD'].values))

    # connect b min and max
    min_KM_b = df_all['KM_b'].min()
    max_KM_b = df_all['KM_b'].max()
    min_KMD_b = df_all.loc[df_all['KM_b'] == min_KM_b, 'KMD_b'].values[0]
    max_KMD_b = df_all.loc[df_all['KM_b'] == max_KM_b, 'KMD_b'].values[0]
    line_b, = ax.plot([min_KM_b, max_KM_b], [min_KMD_b, max_KMD_b], color='orange', visible=False)
    ax.text((min_KM_b + max_KM_b) / 2, (min_KMD_b + max_KMD_b) / 2, 'b', color='black',fontsize=12)

    # connect y min and max
    min_KM_y = df_all['KM_y'].min()
    max_KM_y = df_all['KM_y'].max()
    min_KMD_y = df_all.loc[df_all['KM_y'] == min_KM_y, 'KMD_y'].values[0]
    max_KMD_y = df_all.loc[df_all['KM_y'] == max_KM_y, 'KMD_y'].values[0]
    line_y, = ax.plot([min_KM_y, max_KM_y], [min_KMD_y, max_KMD_y], color='green', visible=False)
    ax.text((min_KM_y + max_KM_y) / 2, (min_KMD_y + max_KMD_y) / 2, 'y', color='black', fontsize=12)

    # limits of the plot
    max_km = max(df_all['KM'].max(), df_all['KM_b'].max(), df_all['KM_y'].max())
    max_kmd = max(df_all['KMD'].max(), df_all['KMD_b'].max(), df_all['KMD_y'].max())
    min_kmd = min(df_all['KMD'].min(), df_all['KMD_b'].min(), df_all['KMD_y'].min())
    ax.set_xlim(0, max_km + 100)
    ax.set_ylim(min_kmd - 0.1, max_kmd + 0.1)

    # set ticks and labels on axes
    xticks = [200 * i for i in range((int(max_km) // 200) + 2)]
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{xtick}" for xtick in xticks])
    ax.set_xlabel('KM')
    ax.set_ylabel('KMD * 1000')
    ax.set_title("KM/KMD plot")
    ax.margins(x=0.05, y=0.05)

    # pan and zoom
    plt.gca().set(xmargin=0.05, ymargin=0.05)
    plt.gca().set_autoscale_on(True)
    plt.gca().autoscale_view()

    # save scatter button
    ax_button = plt.axes([0.85, 0.01, 0.13, 0.05])
    btn = Button(ax_button, 'Save New Scatter')

    # checkboxes
    check_ax = plt.axes([0.79, 0.79, 0.2, 0.2], frameon=True, facecolor='#EDEDED')
    check = CheckButtons(check_ax,['B and y lines' ,'Max. distance', 'Limit 1 area','Limit 1 scatter', 'Limit 2 area', 'Limit 2 scatter'], [False, False, False, False, False, False])

    #maximum distance points from b and y line from their own sets of data
    point_b, distance_b = find_max_dist(df_all, 'KM_b', 'KMD_b')
    annotation_b = ax.annotate(f'maximal distance \n from b line:{distance_b/1000:.4f}', fontsize=10, xy=(point_b),textcoords="offset points", xytext=(-10, 20), ha='center',arrowprops={'arrowstyle': '->'}, visible=False)
    point_y, distance_y = find_max_dist(df_all, 'KM_y', 'KMD_y')
    annotation_y = ax.annotate(f'maximal distance \n from y line:{distance_y/1000:.4f}', fontsize=10, xy=(point_y),textcoords="offset points", xytext=(10, 20), ha='center',arrowprops={'arrowstyle': '->'}, visible=False)

    # 1) parallel lines for b and area
    bx1, by1, bx2, by2 = parallel_line(min_KM_b, min_KMD_b, max_KM_b, max_KMD_b, distance_b)
    line_b_parallel_above, = ax.plot([bx1, bx2], [by1, by2], color='orange', linestyle='dotted', visible=False)
    bx1_below, by1_below, bx2_below, by2_below = parallel_line(min_KM_b, min_KMD_b, max_KM_b, max_KMD_b, -distance_b)
    line_b_parallel_below, = ax.plot([bx1_below, bx2_below], [by1_below, by2_below], color='orange', linestyle='dotted',
                                     visible=False)
    parallelogram_vertices_b = [(bx1, by1), (bx2, by2), (bx2_below, by2_below), (bx1_below, by1_below)]
    parallelogram_b = Polygon(parallelogram_vertices_b, closed=True, facecolor='orange', alpha=0.2, visible=False)
    ax.add_patch(parallelogram_b)

    # 2) parallel lines for b and area
    distance2 = 0.085 * 1000
    bx3, by3, bx4, by4 = parallel_line(min_KM_b, min_KMD_b, max_KM_b, max_KMD_b, distance2)
    line_b_parallel_above2, = ax.plot([bx3, bx4], [by3, by4], color='orange', linestyle='dotted', visible=False)
    bx3_below, by3_below, bx4_below, by4_below = parallel_line(min_KM_b, min_KMD_b, max_KM_b, max_KMD_b, -distance2)
    line_b_parallel_below2, = ax.plot([bx3_below, bx4_below], [by3_below, by4_below], color='orange',
                                      linestyle='dotted', visible=False)
    parallelogram_vertices_b2 = [(bx3, by3), (bx4, by4), (bx4_below, by4_below), (bx3_below, by3_below)]
    parallelogram_b2 = Polygon(parallelogram_vertices_b2, closed=True, facecolor='orange', alpha=0.2, visible=False)
    ax.add_patch(parallelogram_b2)

    # 1) parallel lines for y and area
    yx1, yy1, yx2, yy2 = parallel_line(min_KM_y, min_KMD_y, max_KM_y, max_KMD_y, distance_y)
    line_y_parallel_above, = ax.plot([yx1, yx2], [yy1, yy2], color='green', linestyle='dotted', visible=False)
    yx1_below, yy1_below, yx2_below, yy2_below = parallel_line(min_KM_y, min_KMD_y, max_KM_y, max_KMD_y, -distance_y)
    line_y_parallel_below, = ax.plot([yx1_below, yx2_below], [yy1_below, yy2_below], color='green', linestyle='dotted',
                                     visible=False)
    parallelogram_vertices_y = [(yx1, yy1), (yx2, yy2), (yx2_below, yy2_below), (yx1_below, yy1_below)]
    parallelogram_y = Polygon(parallelogram_vertices_y, closed=True, facecolor='green', alpha=0.2, visible=False)
    ax.add_patch(parallelogram_y)

    # 2) parallel lines for y and area
    yx3, yy3, yx4, yy4 = parallel_line(min_KM_y, min_KMD_y, max_KM_y, max_KMD_y, distance2)
    line_y_parallel_above2, = ax.plot([yx3, yx4], [yy3, yy4], color='green', linestyle='dotted', visible=False)
    yx3_below, yy3_below, yx4_below, yy4_below = parallel_line(min_KM_y, min_KMD_y, max_KM_y, max_KMD_y, -distance2)
    line_y_parallel_below2, = ax.plot([yx3_below, yx4_below], [yy3_below, yy4_below], color='green', linestyle='dotted',
                                      visible=False)
    parallelogram_vertices_y2 = [(yx3, yy3), (yx4, yy4), (yx4_below, yy4_below), (yx3_below, yy3_below)]
    parallelogram_y2 = Polygon(parallelogram_vertices_y2, closed=True, facecolor='green', alpha=0.2, visible=False)
    ax.add_patch(parallelogram_y2)

    # 1) perpendicular lines
    perp_lines_b, points_within_b = perpendicular_lines(min_KM_b, min_KMD_b, max_KM_b, max_KMD_b, distance_b, scatters)
    if points_within_b:
        scatter_b = ax.scatter(*zip(*points_within_b), color='orange', s=3, visible =False)

    perp_lines_y,points_within_y = perpendicular_lines(min_KM_y, min_KMD_y, max_KM_y, max_KMD_y, distance_y, scatters)
    if points_within_y:
        scatter_y= ax.scatter(*zip(*points_within_y), color='green', s=3, visible= False)

    # 2) perpendicular lines
    perp_lines_b2, points_within_b2 = perpendicular_lines(min_KM_b, min_KMD_b, max_KM_b, max_KMD_b, distance2,scatters)
    if points_within_b2:
        scatter_b2 = ax.scatter(*zip(*points_within_b2), color='orange', s=3, visible=False)

    perp_lines_y2, points_within_y2 = perpendicular_lines(min_KM_y, min_KMD_y, max_KM_y, max_KMD_y, distance2,scatters)
    if points_within_y2:
        scatter_y2 = ax.scatter(*zip(*points_within_y2), color='green', s=3, visible=False)

    # 1)new scatter info
    new_scatters = {**points_within_b, **points_within_y}
    total_new_scatters = len(new_scatters)
    total_scatter = len(scatters)
    percentage = (total_new_scatters / total_scatter) * 100 if total_scatter else 0
    text_obj = ax.annotate(f'New Scatter: {total_new_scatters}, Percentage: {percentage:.2f}%', xy=(0.89, 0.8),xycoords='axes fraction',bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="aliceblue"), visible=False)

    # 2) new scatter info
    new_scatters2 = {**points_within_b2, **points_within_y2}
    total_new_scatters2 = len(new_scatters2)
    percentage2 = (total_new_scatters2 / total_scatter) * 100 if total_scatter else 0
    text_obj = ax.annotate(f'New Scatter 2: {total_new_scatters2}, Percentage 2: {percentage2:.2f}%', xy=(0.89, 0.8),xycoords='axes fraction',bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="aliceblue"), visible=False)

    # show cursor only when on dot
    def on_motion(event):
        if event.inaxes != ax:
            for sel in list(cursor.selections):
                cursor.remove_selection(sel)
            fig.canvas.draw_idle()

    # checkbox line visibility
    def update_lines(label):
        nonlocal line_b, line_y
        if label == 'B and y lines':
            line_b.set_visible(not line_b.get_visible())
            line_y.set_visible(not line_y.get_visible())
        elif label == 'Max. distance':
            annotation_b.set_visible(not annotation_b.get_visible())
            annotation_y.set_visible(not annotation_y.get_visible())
        elif label == 'Limit 1 area':
            line_b_parallel_above.set_visible(not line_b_parallel_above.get_visible())
            line_b_parallel_below.set_visible(not line_b_parallel_below.get_visible())
            line_y_parallel_above.set_visible(not line_y_parallel_above.get_visible())
            line_y_parallel_below.set_visible(not line_y_parallel_below.get_visible())
            parallelogram_b.set_visible(not parallelogram_b.get_visible())
            parallelogram_y.set_visible(not parallelogram_y.get_visible())
        elif label == 'Limit 2 area':
            line_b_parallel_above2.set_visible(not line_b_parallel_above2.get_visible())
            line_b_parallel_below2.set_visible(not line_b_parallel_below2.get_visible())
            line_y_parallel_above2.set_visible(not line_y_parallel_above2.get_visible())
            line_y_parallel_below2.set_visible(not line_y_parallel_below2.get_visible())
            parallelogram_b2.set_visible(not parallelogram_b2.get_visible())
            parallelogram_y2.set_visible(not parallelogram_y2.get_visible())
        elif label == 'Limit 1 scatter':
            scatter_b.set_visible(not scatter_b.get_visible())
            scatter_y.set_visible(not scatter_y.get_visible())
            if scatter_b.get_visible() and scatter_y.get_visible():
                total_new_scatters = len({**points_within_b, **points_within_y})
            elif scatter_b.get_visible():
                total_new_scatters = len(points_within_b)
            elif scatter_y.get_visible():
                total_new_scatters = len(points_within_y)
            else:
                total_new_scatters = 0
            total_scatter = len(scatters)
            percentage = (total_new_scatters / total_scatter) * 100 if total_scatter else 0
            text_obj.set_text(f'New Scatter: {total_new_scatters}\nPercentage: {percentage:.2f}%')
            text_obj.set_visible(True if (scatter_b.get_visible() or scatter_y.get_visible()) else False)
        elif label == 'Limit 2 scatter':
            scatter_b2.set_visible(not scatter_b2.get_visible())
            scatter_y2.set_visible(not scatter_y2.get_visible())
            if scatter_b2.get_visible() and scatter_y2.get_visible():
                total_new_scatters2 = len({**points_within_b2, **points_within_y2})
            elif scatter_b2.get_visible():
                total_new_scatters2 = len(points_within_b2)
            elif scatter_y2.get_visible():
                total_new_scatters2 = len(points_within_y2)
            else:
                total_new_scatters2 = 0
            total_scatter = len(scatters)
            percentage2 = (total_new_scatters2 / total_scatter) * 100 if total_scatter else 0
            text_obj.set_text(f'New Scatter 2: {total_new_scatters2}\nPercentage 2: {percentage2:.2f}%')
            text_obj.set_visible(True if (scatter_b2.get_visible() or scatter_y2.get_visible()) else False)
    plt.draw()

    # cursor reacts to dots in the first column
    def cursor_callback(sel):
        hovered_point = (sel.target[0], sel.target[1])
        new_distance_b = point_to_line_distance2(min_KM_b, min_KMD_b, max_KM_b, max_KMD_b, hovered_point[0],
                                              hovered_point[1])
        new_distance_y= point_to_line_distance2(min_KM_y, min_KMD_y, max_KM_y, max_KMD_y, hovered_point[0],
                                              hovered_point[1])
        text = f"KM={sel.target[0]:.4f}, KMD={sel.target[1] / 1000:.4f},\n" f"Mass={df_all['Mass'][sel.index]}, Intensity={df_all['Intensity'][sel.index]}"

        if hovered_point in points_within_b or hovered_point in points_within_b2:
            if new_distance_b is not None:
                text += f"\nDistance={new_distance_b / 1000:.4f}"

        if hovered_point in points_within_y or hovered_point in points_within_y2:
            if new_distance_y is not None:
                text += f"\nDistance={new_distance_y / 1000:.4f}"

        sel.annotation.set_text(text)
        sel.annotation.get_bbox_patch().set(fc="pink")
        sel.annotation.set_fontproperties(font_manager.FontProperties(size=8))

    red_scatter_points = []
    red_annotations = []

    # search
    def submit(text):
        if text:
            try:
                mass_value = float(text)
                if np.any(np.isclose(df_all['Mass'].values, mass_value, atol=0)):
                    km, kmd = convert_single_mass_to_km_kmd(mass_value)
                    red_scatter = ax.scatter([km], [kmd * 1000], color='red', s=13)

                    # Adding annotation
                    annotation = ax.annotate(f'{km:.4f}, {kmd:.4f}', (km, kmd * 1000), textcoords="offset points",
                                             xytext=(0, 10), ha='center', fontsize=8)

                    # Append scatter and annotation to their respective lists
                    red_scatter_points.append(red_scatter)
                    red_annotations.append(annotation)

                    plt.draw()
                else:
                    print("Mass value not found.")
            except ValueError:
                print("Invalid input. Please enter a valid number.")
        else:
            print("Text is empty.")

    def delete(text):
        if text:
            try:
                mass_value = float(text)
                km, kmd = convert_single_mass_to_km_kmd(mass_value)
                for scatter, annotation in zip(red_scatter_points, red_annotations):
                    scatter_data = scatter.get_offsets()
                    for i, point in enumerate(scatter_data):
                        if abs(point[0] - km) < 1e-8 and abs(point[1] - kmd * 1000) < 1e-8:
                            scatter.remove()
                            annotation.remove()
                            red_scatter_points.remove(scatter)
                            red_annotations.remove(annotation)
                            plt.draw()
                            return
            except ValueError:
                print("Invalid input. Please enter a valid number.")
        else:
            print("Text is empty.")

    global distances_b
    global distances_y
    global distances_y2
    global distances_b2
    distances_b = {}
    distances_y = {}
    distances_b2 = {}
    distances_y2 = {}

    for px, py in points_within_b.keys():
        distance_to_main_line = point_to_line_distance2(min_KM_b, min_KMD_b, max_KM_b, max_KMD_b, px, py)
        distances_b[(px, py)] = distance_to_main_line

    for px, py in points_within_b2.keys():
        distance_to_main_line = point_to_line_distance2(min_KM_b, min_KMD_b, max_KM_b, max_KMD_b, px, py)
        distances_b2[(px, py)] = distance_to_main_line

    for px, py in points_within_y.keys():
        distance_to_main_line = point_to_line_distance2(min_KM_y, min_KMD_y, max_KM_y, max_KMD_y, px, py)
        distances_y[(px, py)] = distance_to_main_line

    for px, py in points_within_y2.keys():
        distance_to_main_line = point_to_line_distance2(min_KM_y, min_KMD_y, max_KM_y, max_KMD_y, px, py)
        distances_y2[(px, py)] = distance_to_main_line

    def save_scatter(event):
        global distances_b
        global distances_y
        global distances_y2
        global distances_b2
        just_distances_b = list(distances_b.values())
        just_distances_y = list(distances_y.values())
        just_distances_b2 = list(distances_b2.values())
        just_distances_y2 = list(distances_y2.values())
        filename = easygui.filesavebox(msg='Choose name for scatter file:', default='*.csv')
        if filename:
            append_to_csv(filename, points_within_b, just_distances_b, points_within_y, just_distances_y, points_within_b2, just_distances_b2, points_within_y2, just_distances_y2)

            with open(filename, 'r') as csvfile:
                data = csvfile.read()
                data = data.replace('--', '')

            with open(filename, 'w') as csvfile:
                csvfile.write(data)

    cursor = mplcursors.cursor([all_points], hover=True)
    cursor.connect("add", cursor_callback)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)

    btn.on_clicked(save_scatter)
    check.on_clicked(update_lines)

    text_box = TextBox(plt.axes([0.1, 0.01, 0.2, 0.04]), 'Enter mass:')
    text_box.on_submit(submit)

    delete_box = TextBox(plt.axes([0.4, 0.01, 0.2, 0.04]), 'Delete mass:')
    delete_box.on_submit(delete)

    plt.show(block=True)

if __name__ == "__main__":
    file_all = select_file("Upload file")
    if not file_all:
        exit(0)

    plot_graph(file_all)

#kada radim izračune 'kmd', tj. 'kmd*1000', ali kada prikazujem 'kmd/1000' (annotation, print...) (exceptions: submit, delete, distance2)