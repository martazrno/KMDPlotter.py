import pandas as pd
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

# when a dot on graph is picked
def on_pick(event):
    ind = event.ind[0]
    data = event.artist.get_gid()[ind]

def convert_to_km_kmd(df, column_name):
    df[column_name] = pd.to_numeric(df[column_name], errors='coerce')
    df = df.dropna(subset=[column_name])  # ignore letters
    konstant = 0.99965856
    km_values = df[column_name] * konstant
    kmd_values = -(km_values - round(km_values, 0)) * 1000
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

#find points close to line using lines perpendicular to it
def perpendicular_lines(x1, y1, x2, y2, perp_length, scatter_points):
    points_within = {}
    perp_lines = []
    touched_scatter = {}  # dictionary to store which scatter point is touched by which perp line
    touched_count = 0  # counter for touched scatter points

    original_length = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    num_perp_lines = int(original_length) * 3
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

#find max distance from line
def find_max_distance(df_all):
    max_KM_ion = df_all['KM_ion'].max()
    min_KM_ion = df_all['KM_ion'].min()
    min_KMD_ion = df_all.loc[df_all['KM_ion'] == min_KM_ion, 'KMD_ion'].values[0]
    max_KMD_ion = df_all.loc[df_all['KM_ion'] == max_KM_ion, 'KMD_ion'].values[0]

    distances = abs((max_KM_ion - min_KM_ion) * (min_KMD_ion - df_all['KMD_ion']) - (min_KM_ion - df_all['KM_ion']) * (
            max_KMD_ion - min_KMD_ion)) / (math.sqrt((max_KM_ion - min_KM_ion) ** 2 + (max_KMD_ion - min_KMD_ion) ** 2))

    max_distance_index = distances.idxmax()
    max_distance = distances[max_distance_index]
    max_distance_point = (df_all.loc[max_distance_index, 'KM_ion'], df_all.loc[max_distance_index, 'KMD_ion'])

    return max_distance_point, max_distance

#find distance from extra scatter to line
def find_distance_extra(df_all):
    max_KM_ion = df_all['KM_ion'].max()
    min_KM_ion = df_all['KM_ion'].min()
    min_KMD_ion = df_all.loc[df_all['KM_ion'] == min_KM_ion, 'KMD_ion'].values[0]
    max_KMD_ion = df_all.loc[df_all['KM_ion'] == max_KM_ion, 'KMD_ion'].values[0]

    distances = abs((max_KM_ion - min_KM_ion) * (min_KMD_ion - df_all['KMD_extra']) - (min_KM_ion - df_all['KM_extra']) * (
            max_KMD_ion - min_KMD_ion)) / (math.sqrt((max_KM_ion - min_KM_ion) ** 2 + (max_KMD_ion - min_KMD_ion) ** 2))

    return distances

def find_dist_ion(df_all):
    max_KM_ion = df_all['KM_ion'].max()
    min_KM_ion = df_all['KM_ion'].min()
    min_KMD_ion = df_all.loc[df_all['KM_ion'] == min_KM_ion, 'KMD_ion'].values[0]
    max_KMD_ion = df_all.loc[df_all['KM_ion'] == max_KM_ion, 'KMD_ion'].values[0]

    distances = abs((max_KM_ion - min_KM_ion) * (min_KMD_ion - df_all['KMD_ion']) - (min_KM_ion - df_all['KM_ion']) * (
            max_KMD_ion - min_KMD_ion)) / (math.sqrt((max_KM_ion - min_KM_ion) ** 2 + (max_KMD_ion - min_KMD_ion) ** 2))

    return distances

def format_point(point):
    return "{:.4f}".format(point) if point is not None and not np.ma.is_masked(point) and not math.isnan(point) else ''

def format_distance(distance):
    return "{:.4f}".format(distance) if distance is not None and not math.isnan(distance) else ''

def append_to_csv(file_name, points_1, distances_1, points_2, distances_2, points_3, distances_3, points_4, distances_4):
    points_1 = list(points_1)
    points_2 = list(points_2)
    points_3 = list(points_3)
    points_4 = list(points_4)

    max_length = max(len(points_1), len(distances_1), len(points_2), len(distances_2), len(points_3), len(distances_3), len(points_4), len(distances_4))

    with open(file_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Limit 1 KM', 'Limit 1 KMD', 'Limit 1 distances', '', 'Limit 2 KM', 'Limit 2 KMD', 'Limit 2 distances','', 'Ion KM','Ion KMD','Ion distances','','Extra KM', 'Extra KMD', 'Extra distance'])

        for i in range(max_length):
            row = []

            row.extend([
                format_point(points_1[i][0]) if i < len(points_1) else '',
                format_point(points_1[i][1] / 1000) if i < len(points_1) else '',
                format_distance(distances_1[i] / 1000) if i < len(distances_1) else '',
                '',
                format_point(points_2[i][0]) if i < len(points_2) else '',
                format_point(points_2[i][1] / 1000) if i < len(points_2) else '',
                format_distance(distances_2[i] / 1000) if i < len(distances_2) else '',
                '',
                format_point(points_3[i][0]) if i < len(points_3) else '',
                format_point(points_3[i][1] / 1000) if i < len(points_3) else '',
                format_distance(distances_3[i] / 1000) if i < len(distances_3) else '',
                '',
                format_point(points_4[i][0]) if i < len(points_4) else '',
                format_point(points_4[i][1] / 1000) if i < len(points_4) else '',
                format_distance(distances_4[i] / 1000) if i < len(distances_4) else '',
                ''
            ])

            writer.writerow(row)

def plot_graph():

    file_path = input("Enter the path to your Excel file: ")
    while True:
        # reading from datafile
        df_all = pd.read_excel(file_path, header=None, skiprows=1)
        df_all.columns = ['Mass', 'Intensity', 'Mass_ion', 'Extra']

        # mass to KM and KMD
        plt.style.use('seaborn')
        df_all['KM'], df_all['KMD'] = convert_to_km_kmd(df_all, 'Mass')
        df_all['KM_ion'], df_all['KMD_ion'] = convert_to_km_kmd(df_all, 'Mass_ion')
        df_all['KM_extra'], df_all['KMD_extra'] = convert_to_km_kmd(df_all, 'Extra')
        fig, ax = plt.subplots(figsize=(11, 7))
        ax.set_position([0.1, 0.1, 0.815, 0.82])


        # plot data
        all_points = ax.scatter(df_all['KM'], df_all['KMD'], color='lightblue', s=3)
        ax.scatter(df_all['KM_ion'], df_all['KMD_ion'], color='orange', s=10)
        scatters_extra = ax.scatter(df_all['KM_extra'], df_all['KMD_extra'], color='purple', s=20, visible=False)

        # For picker functionality
        all_points.set_picker(True)
        all_points.set_pickradius(5)
        all_points.set_gid(df_all[['Mass', 'Intensity', 'KM', 'KMD']].values)
        fig.canvas.mpl_connect('pick_event', on_pick)

        # For your perpendicular line calculations
        scatters = list(zip(df_all['KM'].values, df_all['KMD'].values))

        # make a line
        min_KM_ion = df_all['KM_ion'].min()
        max_KM_ion = df_all['KM_ion'].max()
        min_KMD_ion = df_all.loc[df_all['KM_ion'] == min_KM_ion, 'KMD_ion'].values[0]
        max_KMD_ion = df_all.loc[df_all['KM_ion'] == max_KM_ion, 'KMD_ion'].values[0]
        line, = ax.plot([min_KM_ion, max_KM_ion], [min_KMD_ion, max_KMD_ion], color='orange', visible=False)

        # limits of the plot
        max_km = df_all['KM'].max()
        max_kmd = df_all['KMD'].max()
        min_kmd = df_all['KMD'].min()
        ax.set_xlim(0, max_km + 100)
        ax.set_ylim(min_kmd - 0.1, max_kmd + 0.1)

        # set ticks and labels on axes
        xticks = [200 * i for i in range((int(max_km) // 200) + 2)]
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{xtick}" for xtick in xticks])
        ax.set_xlabel('KM', weight='bold', color='gray')
        ax.set_ylabel('KMD * 1000', weight='bold', color='gray')
        ax.set_title("KM/KMD plot", weight='bold', color='magenta')
        ax.margins(x=0.05, y=0.05)

        # pan and zoom
        plt.gca().set(xmargin=0.05, ymargin=0.05)
        plt.gca().set_autoscale_on(True)
        plt.gca().autoscale_view()

        # maximum distance points from line from their own sets of data
        point, distance = find_max_distance(df_all)
        annotation = ax.annotate(f'maximal distance \n from line:{distance / 1000:.4f}', fontsize=10, xy=(point),
                                 textcoords="offset points", xytext=(-10, 20), ha='center', arrowprops={'arrowstyle': '->'},
                                 visible=False)

        # 1) parallel lines for line and area
        x1, y1, x2, y2 = parallel_line(min_KM_ion, min_KMD_ion, max_KM_ion, max_KMD_ion, distance)
        line_parallel_above, = ax.plot([x1, x2], [y1, y2], color='orange', linestyle='dotted', visible=False)
        x1_below, y1_below, x2_below, y2_below = parallel_line(min_KM_ion, min_KMD_ion, max_KM_ion, max_KMD_ion, -distance)
        line_parallel_below, = ax.plot([x1_below, x2_below], [y1_below, y2_below], color='orange', linestyle='dotted',
                                       visible=False)
        parallelogram_vertices = [(x1, y1), (x2, y2), (x2_below, y2_below), (x1_below, y1_below)]
        parallelogram = Polygon(parallelogram_vertices, closed=True, facecolor='orange', alpha=0.2, visible=False)
        ax.add_patch(parallelogram)

        # 2) parallel lines for line and area
        distance2 = 0.085 * 1000
        x3, y3, x4, y4 = parallel_line(min_KM_ion, min_KMD_ion, max_KM_ion, max_KMD_ion, distance2)
        line_parallel_above2, = ax.plot([x3, x4], [y3, y4], color='orange', linestyle='dotted', visible=False)
        x3_below, y3_below, x4_below, y4_below = parallel_line(min_KM_ion, min_KMD_ion, max_KM_ion, max_KMD_ion, -distance2)
        line_parallel_below2, = ax.plot([x3_below, x4_below], [y3_below, y4_below], color='orange',
                                        linestyle='dotted', visible=False)
        parallelogram_vertices_2 = [(x3, y3), (x4, y4), (x4_below, y4_below), (x3_below, y3_below)]
        parallelogram_2 = Polygon(parallelogram_vertices_2, closed=True, facecolor='orange', alpha=0.2, visible=False)
        ax.add_patch(parallelogram_2)

        # 1) perpendicular lines
        perp_lines, points_within = perpendicular_lines(min_KM_ion, min_KMD_ion, max_KM_ion, max_KMD_ion, distance,scatters)
        if points_within:
            scatter = ax.scatter(*zip(*points_within), color='orange', s=3, visible=False)

        # 2) perpendicular lines
        perp_lines_2, points_within_2 = perpendicular_lines(min_KM_ion, min_KMD_ion, max_KM_ion, max_KMD_ion, distance2,scatters)
        if points_within_2:
            scatter_2 = ax.scatter(*zip(*points_within_2), color='orange', s=3, visible=False)

        # 1)new scatter info
        total_new_scatters = len(points_within)
        total_scatter = len(scatters)
        percentage = (total_new_scatters / total_scatter) * 100 if total_scatter else 0
        text_obj = ax.annotate(f'New Scatter: {total_new_scatters}, Percentage: {percentage:.2f}%', xy=(0.9, 0.9),
                               xycoords='axes fraction',
                               bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="aliceblue"), visible=False)

        # 2) new scatter info
        total_new_scatters2 = len(points_within_2)
        percentage2 = (total_new_scatters2 / total_scatter) * 100 if total_scatter else 0
        text_obj = ax.annotate(f'New Scatter 2: {total_new_scatters2}, Percentage 2: {percentage2:.2f}%', xy=(0.9, 0.9),
                               xycoords='axes fraction',
                               bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="aliceblue"), visible=False)

        #save scatter
        extra_points = scatters_extra.get_offsets()
        ion_scatter = (ax.scatter(df_all['KM_ion'], df_all['KMD_ion'], color='orange', s=6)).get_offsets()
        global distances_1
        global distances_2
        distances_1 = {}
        distances_2 = {}
        for px, py in points_within.keys():
            distance_to_main_line = point_to_line_distance2(min_KM_ion, min_KMD_ion, max_KM_ion, max_KMD_ion, px, py)
            distances_1[(px, py)] = distance_to_main_line
        for px, py in points_within_2.keys():
            distance_to_main_line = point_to_line_distance2(min_KM_ion, min_KMD_ion, max_KM_ion, max_KMD_ion, px, py)
            distances_2[(px, py)] = distance_to_main_line
        def save_scatter():
            global distances_1
            global distances_2
            just_distances_1 = list(distances_1.values())
            just_distances_2 = list(distances_2.values())
            ion_distances = find_dist_ion(df_all)
            extra_distances = find_distance_extra(df_all)
            file_name = input("Enter the name of the file to save with: ")
            file_name += ".csv"

            if file_name:
                append_to_csv(file_name, points_within, just_distances_1, points_within_2, just_distances_2, ion_scatter,
                              ion_distances, extra_points, extra_distances)

                with open(file_name, 'r') as csvfile:
                    data = csvfile.read()
                    data = data.replace('--', '')

                with open(file_name, 'w') as csvfile:
                    csvfile.write(data)

        def update_lines(choice):
            nonlocal line
            if choice == '1':
                line.set_visible(not line.get_visible())
            elif choice == '2':
                scatters_extra.set_visible(not scatters_extra.get_visible())
            elif choice == '3':
                annotation.set_visible(not annotation.get_visible())
            elif choice == '4':
                line_parallel_above.set_visible(not line_parallel_above.get_visible())
                line_parallel_below.set_visible(not line_parallel_below.get_visible())
                parallelogram.set_visible(not parallelogram.get_visible())
            elif choice == '5':
                parallelogram_2.set_visible(not parallelogram_2.get_visible())
                line_parallel_above2.set_visible(not line_parallel_above2.get_visible())
                line_parallel_below2.set_visible(not line_parallel_below2.get_visible())
            elif choice == '6':
                scatter.set_visible(not scatter.get_visible())
                if scatter.get_visible():
                    total_new_scatters = len(points_within)
                else:
                    total_new_scatters = 0
                total_scatter = len(scatters)
                percentage = (total_new_scatters / total_scatter) * 100 if total_scatter else 0
                text_obj.set_text(f'New Scatter: {total_new_scatters}\nPercentage: {percentage:.2f}%')
                text_obj.set_visible(True if (scatter.get_visible()) else False)
            elif choice == '7':
                scatter_2.set_visible(not scatter_2.get_visible())
                if scatter_2.get_visible():
                    total_new_scatters2 = len(points_within_2)
                else:
                    total_new_scatters2 = 0
                total_scatter = len(scatters)
                percentage2 = (total_new_scatters2 / total_scatter) * 100 if total_scatter else 0
                text_obj.set_text(f'New Scatter 2: {total_new_scatters2}\nPercentage 2: {percentage2:.2f}%')
                text_obj.set_visible(True if (scatter_2.get_visible()) else False)
            elif choice == '9':
                exit()

        user_choices = input("Enter your choices: 1.Ion line, 2.Extra scatter, 3.Max.distance, 4.Limit 1 area, 5.Limit 2 area, 6.Limit 1 scatter, 7.Limit 2 scatter, 8. Save scatter, 9.Exit:")
        choices = user_choices.split(',')
        current_choices = []

        for user_choice in choices:
            user_choice = user_choice.strip()
            if user_choice in ['1', '2', '3', '4', '5', '6', '7', '8', '9']:
                update_lines(user_choice)
                current_choices.append(user_choice)

        if '8' in current_choices:
            save_scatter()
            if len(current_choices) == 1:
                continue

        if any(choice in current_choices for choice in ['1', '2', '3', '4', '5', '6', '7']):
            plt.savefig(f'KMD_graph_{"_".join([c for c in current_choices if c != "8"])}.png')
            plt.draw()
            plt.show()

if __name__ == "__main__":
    plot_graph()

''' 
Kad se pokrene program, traži se od korisnika da upiše path za excel file. Nakon toga slijedi input za interakciju s grafom.
Nakon svakog inputa radi se nova slika ili/i csv file (ovisno o odabiru). Može odjednom biti više izbora, no moraju se
odvojiti zarezom, npr.input: 1,2,3. Iz programa se izlazi odabirom 9, i to mora tada biti jedini izbor. Svakim odabirom prikaže 
se nova slika, i spremi se zajedno s brojevima odabira u nazivu, npr. ako su odabiri 2,3,5 onda će se slika zvati KMD_graph_2_3_5.png. 
Ako je odabir samo 8, treba pričekati minutu dok se csv file ne spremi, no ako je odabir npr. 2,3,8, ne treba se čekati za csv file. 
Također ne treba se čekati ako se prekine program. Ne znam zašto je to tako... Nakon stvaranje slike, ako želimo upisati 
nove odabire mora se slika ugasiti.
'''