#!/usr/bin/env python3
import os
import time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import imageio
import glob
from PIL import Image


class analysis:

    def __init__(self):
        # initialize parameters
        self.analysis_dir = 'gen_data'
        # todo change these values
        self.human_set_loss_error = 0.1
        self.human_perseverative_error = 0.1
        self.models_chart_set_loss = []
        self.models_chart_perseverative = []
        self.models_names = []
        self.number_of_cumulative_steps = 100

    def get_set_loss_error(self, data_frame):

        d = data_frame[["current_reward", "agent_rule"]].to_numpy(dtype=int)

        repeat_count = 0
        set_loss_error = 0

        for i in range(1, d.shape[0]):

            if d[i - 1][0] == 1:
                repeat_count += 1

                if d[i - 1][1] != d[i][1]:
                    set_loss_error += 1

        set_loss_error = set_loss_error / repeat_count

        return set_loss_error

    def get_perseverative_error(self, data_frame):

        d = data_frame[["current_reward", "agent_rule"]].to_numpy(dtype=int)
        switch_count = 0
        perseverative_error = 0

        for i in range(1, d.shape[0]):

            if d[i - 1][0] == -1:
                switch_count += 1

                if d[i - 1][1] == d[i][1]:
                    perseverative_error += 1

        perseverative_error = perseverative_error / switch_count

        return perseverative_error

    def plot_iri_distribution_learning(self, data_frame, limit, step, path):

        # print(df)
        d = data_frame["current_reward"].to_numpy(dtype=int)

        li = []
        streak = 0

        for i in range(limit - step, limit):

            if d[i] == 1:
                streak += 1
            elif d[i] == -1 and d[i - 1] == 1:

                li.append(streak)
                streak = 0

        plt.hist(li)
        plt.axvline(sum(li) / len(li), color="r", linestyle="dashed", linewidth=1)
        plt.xlabel(f'{limit + 1}', fontweight='bold', fontsize=15)
        # save the image in the folder of that csv to create Gif later after finish all steps.
        plt.savefig(f'{path}/{limit + 1}.png')

    def get_all_analysis(self):
        # first we build the Gifs

        # add human values to bar chart
        self.models_chart_set_loss.append(self.human_set_loss_error)
        self.models_chart_perseverative.append(self.human_perseverative_error)
        self.models_names.append("Human")

        # loop on all csv in the analysis directory

        for file in os.listdir(self.analysis_dir):
            if file.endswith('.csv'):
                # todo we can add any model if we need more.

                # add models names to bar chart
                print(file)
                if file.find('vani'):
                    self.models_names.append('vanilla_q')
                else:
                    self.models_names.append('deep_q')
                data = pd.read_csv(f'{self.analysis_dir}/{file}')
                imgs_list = []

                # create folders with same csv name to gather all images of that csv to create Gif
                new_path = f'{self.analysis_dir}/{file[:-4]}'

                if not os.path.isdir(new_path):
                    os.mkdir(new_path)

                # todo we can change the number of cumulative steps.

                # divide the cumulative into steps

                for i in range(len(data.index)):
                    if (i + 1) % self.number_of_cumulative_steps == 0:
                        self.plot_iri_distribution_learning(data, i, self.number_of_cumulative_steps, new_path)
                        # clear the plt to avoid add all plots above each other
                        plt.clf()
                        imgs_list.append(i)

                fp_out = f"{new_path}.gif"

                # create Gif for each csv
                img, *imgs = [Image.open(f'{new_path}/{f + 1}.png') for f in imgs_list]
                img.save(fp=fp_out, format='GIF', append_images=imgs,
                         save_all=True, duration=350, loop=0)

                self.models_chart_set_loss.append(self.get_set_loss_error(data))
                self.models_chart_perseverative.append(self.get_perseverative_error(data))

        # second create the bar chart to compare all models performances.

        plt.clf()
        barWidth = 0.25
        fig = plt.subplots(figsize=(12, 8))
        br1 = np.arange(len(self.models_chart_set_loss))
        br2 = [x + barWidth for x in br1]

        plt.bar(br1, self.models_chart_set_loss, color='r', width=barWidth,
                edgecolor='grey', label='Set Loss Error')
        plt.bar(br2, self.models_chart_perseverative, color='g', width=barWidth,
                edgecolor='grey', label='Preservation Error')

        plt.xlabel('Errors', fontweight='bold', fontsize=15)
        plt.ylabel('Values', fontweight='bold', fontsize=15)
        plt.xticks([r + barWidth for r in range(len(self.models_chart_set_loss))],
                   self.models_names)
        plt.legend()
        plt.savefig(f'{self.analysis_dir}/bar_chart.png')


# run these lines to create the Gifs and bar chart.

an = analysis()
an.get_all_analysis()
