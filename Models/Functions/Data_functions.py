"""Load Files"""
from typing import List, Any

import numpy as np
import matplotlib.pylab as plt
import torch


def import_data(filenames, max_N=3, shape="random"):
    """Open Files"""
    b20 = []
    for i, filename in enumerate(filenames):
        if i == 0:
            b20 = np.loadtxt("./Data/b20_artillery" + filename + ".txt", dtype=float)
            b20_norm = np.loadtxt("./Data/norm_b20_artillery" + filename + ".txt", dtype=float)
            truths = np.loadtxt("./Data/truths_artillery" + filename + ".txt", dtype=float)
        else:
            b20 = np.concatenate((b20, np.loadtxt("./Data/b20_artillery" + filename + ".txt", dtype=float)))
            b20_norm = np.concatenate(
                (b20_norm, np.loadtxt("./Data/norm_b20_artillery" + filename + ".txt", dtype=float)))
            truths = np.concatenate((truths, np.loadtxt("./Data/truths_artillery" + filename + ".txt", dtype=float)))

    if len(b20) != len(truths):
        raise Exception("Arrays not of equal length")

    """Format Data"""
    """Setup Data"""
    b15_norm = [np.concatenate((b[0:3], b[4:7], b[8:11], b[12:15], b[16:19])) for b in
                b20_norm]  # We only care about 1,2,3 (xyz) not 4 (T) per sensor
    b15 = [np.concatenate((b[0:3], b[4:7], b[8:11], b[12:15], b[16:19])) for b in b20]
    # Setup normalization
    norm_val = []
    for i in range(len(b15_norm[0])):
        mean = 0
        for count, b in enumerate(b15_norm):
            mean += b[i]
        mean = mean / count
        norm_val.append(mean)

    b15 = [b / norm_val for b in b15]
    test_truths = [tr[2] for tr in truths]

    """Cleanup Data"""
    b15 = [b15[i] for i, t in enumerate(test_truths) if t < max_N]
    truths = [truths[i] for i, t in enumerate(test_truths) if t < max_N]
    test_truths = [tr[2] for tr in truths]

    if shape == "random":
        truths = [truths[i] if t > 0 else [np.random.randint(3, 17, 1)[0], np.random.randint(3, 17, 1)[0], 0] for i, t
                  in enumerate(test_truths)]
    else:
        truths = [truths[i] if t > 0 else [shape, shape, 0] for i, t in enumerate(test_truths)]
    test_truths = [tr[2] for tr in truths]

    print("Filenames: ", filenames)
    print("No of samples: ", len(b15))
    print("b15[0]:", b15[0])
    print(f"Of which {len(np.where(np.array(test_truths) == 0)[0])} have 0N")
    print(f"Shape of 0N truths: {truths[np.where(np.array(test_truths) == 0)[0][0]]}")
    print("------------------------------------------")
    print("")

    """Final Check"""
    if (len(b15) != len(truths)) or len(b15[0]) != 15 or len(truths[0]) != 3:
        raise Exception("Arrays not of equal length")

    return b15, truths, test_truths, norm_val


def visualize(b15, test_truths):
    x = [[b[0] for b in b15], [b[3] for b in b15], [b[6] for b in b15], [b[9] for b in b15], [b[12] for b in b15]]
    y = [[b[1] for b in b15], [b[4] for b in b15], [b[7] for b in b15], [b[10] for b in b15], [b[13] for b in b15]]
    z = [[b[2] for b in b15], [b[5] for b in b15], [b[8] for b in b15], [b[11] for b in b15], [b[14] for b in b15]]

    """Plots sensor data against truths"""

    fig, axs = plt.subplots(4, 1, sharex=True)
    fig.suptitle('Total Data')

    axs[0].plot(test_truths, ",")
    axs[0].set_title("Truths")
    axs[0].set_ylabel("Force [N]")
    axs[1].set_ylabel("X Sensor data")
    axs[2].set_ylabel("Y Sensor data")
    axs[3].set_ylabel("Z Sensor data")
    axs[3].set_xlabel("Samples")
    for x_n in x:
        axs[1].plot(x_n, ",")
    for y_n in y:
        axs[2].plot(y_n, ",")
    for i, z_n in enumerate(z):
        axs[3].plot(z_n, ",", label="S" + str(i+1))
    plt.legend(loc='upper right', prop={'size': 5})
    plt.show()

    """plot subset of the data"""
    fig = plt.figure()
    gs = fig.add_gridspec(4, 3, hspace=0.01, wspace=0.01)
    axs = gs.subplots(sharex='col', sharey='row')

    fig.suptitle('Subsets')
    loc = int(len(test_truths) / 2)
    subset = 100
    axs[0][0].set_title(f"50 : {subset}")
    axs[0][1].set_title(f"{loc} : {loc + subset}")
    axs[0][2].set_title(f"-50 : {len(test_truths)}")
    axs[0][0].plot(test_truths[50:50 + subset])
    axs[0][1].plot(test_truths[loc:loc + subset])
    axs[0][2].plot(test_truths[-subset:])
    for n in x:
        axs[1][0].plot(n[50:50 + subset])
        axs[1][1].plot(n[loc:loc + subset])
        axs[1][2].plot(n[-subset:])
    """
    for ax in axs[1:]:
        for a in ax:
            a.set_ylim(-8, 8)
    for a in axs[0]:
        a.set_ylim(-1, 5)
    """
    for n in y:
        axs[2][0].plot(n[50:50 + subset])
        axs[2][1].plot(n[loc:loc + subset])
        axs[2][2].plot(n[-subset:])
    for i, n in enumerate(z):
        axs[3][0].plot(n[50:50 + subset])
        axs[3][1].plot(n[loc:loc + subset])
        axs[3][2].plot(n[-subset:], label="S" + str(i+1))

    axs[0][0].set_ylabel("Force [N]")
    axs[1][0].set_ylabel("X")
    axs[2][0].set_ylabel("Y")
    axs[3][0].set_ylabel("Z")
    axs[3][1].set_xlabel("Samples")
    plt.legend(loc='upper right', prop={'size': 5})
    plt.show()

def evaluate_MLP1(model,test_dataset, title = ""):
    print(f"Testing Eval on {len(test_dataset)} testing samples")
    model.eval()
    x_diff = []
    y_diff = []
    F_diff = []
    x_corr = 0
    y_corr = 0
    F_corr = 0
    total_corr = 0

    with torch.no_grad():
        check = len(test_dataset)
        for i in range(check):
            label = test_dataset[i][1].numpy()
            output = model(test_dataset[i][0]).numpy()
            x_diff.append(abs(label[0] - output[0]))
            if x_diff[-1] < 1:
                x_corr += 1
            y_diff.append(abs(label[1] - output[1]))
            if y_diff[-1] < 1:
                y_corr += 1
            F_diff.append(abs(label[2] - output[2]))
            if F_diff[-1] < 0.1:
                F_corr += 1
            if x_diff[-1] < 1 and y_diff[-1] < 1 and F_diff[-1] < 0.1:
                total_corr += 1
    x2 = [x ** 2 for x in x_diff]
    y2 = [x ** 2 for x in y_diff]
    F2 = [x ** 2 for x in F_diff]
    print(" Mean Dist | MSE  | Stdv")
    print(f"X   {np.mean(x_diff):.2f}mm | {np.mean(x2):.2f} | {np.std(x_diff):.2f}")
    print(f"Y   {np.mean(y_diff):.2f}mm | {np.mean(y2):.2f} | {np.std(y_diff):.2f}")
    print(f"F   {np.mean(F_diff):.2f}N  | {np.mean(F2):.2f} | {np.std(F_diff):.2f}")
    print("-------------------------\n")
    print(f"X 1mm accuracy: {(x_corr / check) * 100}%")
    print(f"Y 1mm accuracy: {(y_corr / check) * 100}%")
    print(f"F 0.1N accuracy: {(F_corr / check) * 100}%")
    print(f"Complete accuracy {total_corr / check * 100}%")
    plt.boxplot([x_diff, y_diff, F_diff])
    plt.xticks([1, 2, 3], ["X", "Y", "Z"])
    plt.title(title + "Boxplot of distance to Reality")
    plt.show()

    """Plot Distances"""
    colors = ["r", "g", "b", "y", "m"]
    fin = 0

    while fin < 5:
        i = np.random.randint(0,500,1)[0]
        label = test_dataset[i][1].numpy()
        output = model(test_dataset[i][0]).detach().numpy()

        if label[2] != 0:
            print(f" Plotted: Output: {output}, Label: {label}")
            plt.plot(label[0], label[1], "x", markersize=(label[2]*10) * 1.2,
                     label=str(label[2]) + "N == " + str(output[2]) + "N", color=colors[fin])
            plt.plot(output[0], output[1], "o", markersize=output[2]*10, color=colors[fin])
            fin += 1
    fin = 0
    while fin < 5:
        i += 1
        label = test_dataset[i][1].numpy()
        output = model(test_dataset[i][0]).detach().numpy()
        if label[2] == 0:
            print(f" 0N: Output: {output}, Label: {label}")
            fin += 1

    plt.xlim(0, 20)
    plt.ylim(0, 20)
    plt.legend(prop={'size': 8})
    plt.title(title+" model accuracy")
    plt.show()

def evaluate_MLP(model,test_dataset, title = ""):
    print(f"Testing Eval on {len(test_dataset)} testing samples")
    model.eval()
    x_diff = []
    y_diff = []
    F_diff = []
    x_corr = 0
    y_corr = 0
    F_corr = 0
    total_corr = 0

    with torch.no_grad():
        check = len(test_dataset)
        for i in range(check):
            label = test_dataset[i][1].numpy()
            output = model(test_dataset[i][0]).numpy()
            x_diff.append(abs(label[0] - output[0]))
            if x_diff[-1] < 1:
                x_corr += 1
            y_diff.append(abs(label[1] - output[1]))
            if y_diff[-1] < 1:
                y_corr += 1
            F_diff.append(abs(label[2] - output[2]))
            if F_diff[-1] < 0.1:
                F_corr += 1
            if x_diff[-1] < 1 and y_diff[-1] < 1 and F_diff[-1] < 0.1:
                total_corr += 1
    x2 = [x ** 2 for x in x_diff]
    y2 = [x ** 2 for x in y_diff]
    F2 = [x ** 2 for x in F_diff]


    """Plot Distances"""
    colors = ["r", "g", "b", "y", "m", "c", "k", "burlywood"]
    fin = 0
    fig, (diff, acc) = plt.subplots(1, 2,gridspec_kw={'width_ratios': [1, 2]})
    while fin < 8:
        i = np.random.randint(0,500,1)[0]
        label = test_dataset[i][1].numpy()
        output = model(test_dataset[i][0]).detach().numpy()

        if label[2] != 0:
            print(f" Plotted: Output: {output}, Label: {label}")
            acc.plot(label[0], label[1], "x", label=str(label[2]) + "N == " + str(output[2]) + "N",
                     markersize=(label[2]*10) * 1.2, color=colors[fin])
            acc.plot(output[0], output[1], "o",
                     markersize=output[2]*10, markerfacecolor='none', markeredgecolor=colors[fin])
            fin += 1
    fin = 0
    while fin < 5:
        i += 1
        label = test_dataset[i][1].numpy()
        output = model(test_dataset[i][0]).detach().numpy()
        if label[2] == 0:
            print(f" 0N: Output: {output}, Label: {label}")
            fin += 1
    print("\n\n\n")
    print(" Mean Dist | MSE  | Stdv")
    print(f"X   {np.mean(x_diff):.2f}mm | {np.mean(x2):.2f} | {np.std(x_diff):.2f}")
    print(f"Y   {np.mean(y_diff):.2f}mm | {np.mean(y2):.2f} | {np.std(y_diff):.2f}")
    print(f"F   {np.mean(F_diff):.2f}N  | {np.mean(F2):.2f} | {np.std(F_diff):.2f}")
    print("-------------------------\n")
    print(f"X 1mm accuracy: {(x_corr / check) * 100}%")
    print(f"Y 1mm accuracy: {(y_corr / check) * 100}%")
    print(f"F 0.1N accuracy: {(F_corr / check) * 100}%")
    print(f"Complete accuracy {total_corr / check * 100}%")


    diff.boxplot([x_diff, y_diff, F_diff])
    diff.set_xticks([1, 2, 3], ["X", "Y", "Z"])
    diff.set_title(title + " Boxplot of distance to Reality")

    acc.set_xlim(0, 20)
    acc.set_ylim(0, 20)
    acc.set_title(title+" Model Accuracy")
    plt.legend(prop={'size': 8})
    plt.show()
