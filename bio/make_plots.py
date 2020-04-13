import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math
import pickle
import pandas as pd
import seaborn as sns

#task = "masking"
task = "contextpred"

pth_node = [
    task + '_0.25',
    task + '_0.5',
    task + '_0.75',
    task + '_0.95',
    task + '_0.99',
    task,
]

pth_graph = [
    'supervised_' + task + '_0.25',
    'supervised_' + task + '_0.5',
    'supervised_' + task + '_0.75',
    'supervised_' + task + '_0.95',
    'supervised_' + task + '_0.99',
    'supervised_' + task
]

def gaussians(node=True):
    if node is True:
        # Node level
        mu = [
            0.938,
            0.919,
            0.931,
            0.930,
            0.914,
            0.883
        ]

        sigma = [
            0.0528,
            0.0679,
            0.0631,
            0.0637,
            0.0736,
            0.0857
        ]

        labels = [
            "masking.pth",
            "masking_0.25.pth",
            "masking_0.5.pth",
            "masking_0.75.pth",
            "masking_0.95.pth",
            "masking_0.99.pth"
        ]

        title = "Node-level model on node-level task"

    else:
        # Graph level
        mu = [
            0.686,
            0.788,
            0.799,
            0.770,
            0.772,
            0.809
        ]

        sigma = [
            0.1895,
            0.1364,
            0.1401,
            0.1512,
            0.1475,
            0.1380
        ]

        labels = [
            "supervised_masking.pth",
            "supervised_masking_0.25.pth",
            "supervised_masking_0.5.pth",
            "supervised_masking_0.75.pth",
            "supervised_masking_0.95.pth",
            "supervised_masking_0.99.pth"
        ]

        title = "Graph-level model on node-level task"



    for i in range(len(mu)):
        #x = np.linspace(mu[i] - 3*sigma[i], mu[i] + 3*sigma[i], 100)
        x = np.linspace(0, 1, 200)
        plt.plot(x, stats.norm.pdf(x, mu[i], sigma[i]))

    plt.legend(labels, loc='upper left')
    plt.title(title)
    plt.xlabel("Test accuracy distributions")
    plt.show()


def clean_header(p):
    return p.replace("tmp/", "").replace(".pth","").replace("model_gin/","")


def boxplots(df, labels, adjust, title):
    
    axs = df.boxplot(showfliers=False)

    axs.set_title(title)

    xticks, labels = plt.xticks()
    labels = [str(i+1) + ": " + labels[i]._text for i in range(len(labels))]
    plt.xticks(ticks=xticks, labels=["1", "2", "3", "4", "5", "6"])

    plt.legend(labels, handlelength=0, bbox_to_anchor=(1.05, 1))
    plt.gcf().subplots_adjust(right=adjust)
    plt.show()



if __name__ == "__main__":
    with open("results/" + task + "_stats.pkl", "rb") as handle:
        results = pickle.load(handle)

    data = pd.DataFrame(results)

    data.rename(columns=lambda x: clean_header(x), inplace=True)

    data_node = data[pth_node]
    data_graph = data[pth_graph]

    boxplots(data_node, pth_node, 0.75, "Node-level test accuracy for node-level task (n=456)")
    boxplots(data_graph, pth_graph, 0.68, "Node-level test accuracy for graph-level task (n=456)")