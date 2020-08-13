import json
import os
import matplotlib.pyplot as plt

def get_data(file_path):     
    fp = open(file_path, 'r')
    hist = json.load(fp)

    result = []
    for i in range(len(hist)):
        arch = hist[i]['arch']
        if(hist[i]['data'] != None):
            accuracy = hist[i]['data']['validation_accuracy']
            test_accuracy = hist[i]['data']['test_accuracy']
            cost = hist[i]['data']['training_time']
            result.append((cost, accuracy, test_accuracy, arch))
        else:
            accuracy = 0
            test_accuracy = 0
            cost = 200
            result.append((cost, accuracy, test_accuracy, arch))

    return result

def preprocess(data, mx):
    best = [0.]
    hist = [0.]
    cost = [0.]
    best_arch = []
    best_ever = 0.
    worst_arch = []
    worst_ever = 1.
    aggt_cost = 0.
    best_acc = 0.
    for i in range(len(data)):
        if i > mx:
            break
        aggt_cost += data[i][0]
        acc = data[i][1]
        best_arch = data[i][3] if data[i][1] > best_ever else best_arch
        best_ever = data[i][1] if data[i][1] > best_ever else best_ever

        if data[i][1] > 0:
            worst_arch = data[i][3] if data[i][1] < worst_ever else worst_arch
            worst_ever = data[i][1] if data[i][1] < worst_ever else worst_ever

        best_acc = data[i][1] if data[i][1] > best_acc else best_acc
        hist.append(acc)
        best.append(best_acc)
        cost.append(aggt_cost)
    
    return cost, hist, best, best_arch, best_ever, worst_arch, worst_ever

def plot_history(hist_files, report_path, mx, one_fig=False, name=''):
    plt.figure(figsize=(18, 5))

    n = len(hist_files)
    nrow = (n / 2) + (n % 2)
    ncol = 1
    if n > 1:
        ncol = 2
    ind = 1

    avgs = []

    aggt_cost = 0.
    
    for hist_file, title in hist_files:
        # name = hist_file.replace('./data/', '').replace('/state_history.json', '').replace('/history.json', '')


        history = get_data(hist_file)
        cost, hist, best, best_arch, best_ever, worst_arch, worst_ever = preprocess(history, 100)
        # print(best_arch)
        # print(best_ever)
        # print()
        # print(worst_arch)
        print(worst_ever)
        print()
        aggt_cost += cost[len(cost)-1]

        for h in hist:
            if(h < 0.63):
                continue
            avgs.append(h)

        if not one_fig:
            plt.subplot(nrow, ncol, ind)

        ind += 1

        plt.plot(hist[:mx], 'r.', label='individual', alpha=0.5)
        plt.plot(best[:mx], label='best', color='blue', alpha=0.5, linewidth=2)

        plt.ylabel('accuracy', fontsize=17)
        plt.xlabel('generation', fontsize=17)

        plt.ylim(0.0, 0.7)
        plt.grid()

        if not one_fig:
            plt.title(title + ' Search trajectories (red=individual, blue=best)', fontsize=17)
        else:
            plt.title(name + ' Search trajectories (red=individual, blue=best)', fontsize=17)

        if not one_fig:
            plt.text(50, 0.1, 'Best = %.4f' % (max(hist)), style='italic', bbox={'facecolor': 'w', 'alpha': 0.5}, fontsize=17)
        else:
            plt.text(50, (len(hist_files) - ind + 2) * 0.05, title +' Best = %.4f' % (max(hist)), style='italic', bbox={'facecolor': 'w', 'alpha': 0.5}, fontsize=17)
    
    print("name %s average %.4f, cost = %.2f" % (name, sum(avgs)/len(avgs), (aggt_cost/60)/60))

    plt.savefig(report_path, bbox_inches = 'tight', pad_inches = 0.2, dpi=300)
    # plt.show()

f = [
    ['./data/ga-c-nom/02/state_history.json', 'GA-2'],
    ['./data/ga/01/state_history.json', 'GA-1'],
    ['./data/ga-c-nom/is1/state_history.json', 'GA-2 is1'],
    ['./data/ga-c-nom/is2/state_history.json', 'GA-2 is2'],
    ['./data/ga-c-nom/is3/state_history.json', 'GA-2 is3'],
    ['./data/ga-c-nom/is4/state_history.json', 'GA-2 is4'],
    ['./data/re/03/state_history.json', 'RE'],
]
# plot_history(f, './report/one.png', 100)

f = [
    './data/ga-c-nom/01/state_history.json',
    './data/ga-c-nom/02/state_history.json',
    './data/re/01/state_history.json',
    './data/re/02/state_history.json',
    './data/re/03/state_history.json',
    './data/ga/01/state_history.json'
]
# plot_history(f, './report/one.png', 100)


f = [
    ['./data/ga-isl/01/his_is1.json', 'Isl-GA-1 (1)'],
    ['./data/ga-isl/01/his_is2.json', 'Isl-GA-1 (2)'],
    # ['./data/ga-isl/02/his_is1.json', 'GA-1-i1 (02)'],
    # ['./data/ga-isl/02/his_is2.json', 'GA-1-i2 (02)']
]
# plot_history(f, './report/ga-2-2.png', 200, True, '2-Island-based GA-1')

f = [
    ['./data/ga-c/is1/state_history.json', 'Isl-GA-2 (1)'],
    ['./data/ga-c/is2/state_history.json', 'Isl-GA-2 (2)']
]
# plot_history(f, './report/ga-2-2.png', 200, True, '2-Island-based GA-2')

f = [
    './data/ga-re-a/is1/history.json',
    './data/ga-re-a/is2/history.json',
    './data/ga-re-a/re3/history.json',
    './data/ga-re-a/is4/history.json',
    './data/ga-re-a/is5/history.json',
    './data/ga-re-a/re6/history.json'
]
# plot_history(f, './report/ga-1-re-a.png', 200)

f = [
    ['./data/ga-re-b/is1/history.json', 'Isl-GA-1 (1)'],
    ['./data/ga-re-b/is2/history.json', 'Isl-GA-1 (2)'],
    ['./data/ga-re-b/re3/state_history.json', 'Isl-RE (1)'],
    ['./data/ga-re-b/is4/history.json', 'Isl-GA-1 (3)'],
    ['./data/ga-re-b/is5/history.json', 'Isl-GA-1 (4)'],
    ['./data/ga-re-b/re6/state_history.json', 'Isl-RE (2)']
]
# plot_history(f, './report/ga-1-re-b.png', 200, True, 'Island-based GA-RE')






f = [
    ['./data/ga-isl-a/his_is1.json', 'Isl-GA-1 (1)'], # 0.63
    ['./data/ga-isl-a/his_is2.json', 'Isl-GA-1 (2)'], # 0.64
    ['./data/ga-isl-a/his_is3.json', 'Isl-GA-1 (3)'], # 0.63
    ['./data/ga-isl-a/his_is4.json', 'Isl-GA-1 (4)'], # 0.65
]
# plot_history(f, './report/ga-1-isl-a.png', 200, True, 'Island-based GA')

f = [
    ['./data/ga-isl-b/his_is1.json', 'Isl-GA-1 (1)'], # 0.68
    ['./data/ga-isl-b/his_is2.json', 'Isl-GA-1 (2)'], # 0.65
    ['./data/ga-isl-b/his_is3.json', 'Isl-GA-1 (3)'], # 0.68
    ['./data/ga-isl-b/his_is4.json', 'Isl-GA-1 (4)'], # 0.68
]
# plot_history(f, './report/ga-1-re-b.png', 200, True, 'Island-based GA')


f = [
    ['./data/re/01/state_history.json', 'Isl-1'], # 0.66
    ['./data/re/02/state_history.json', 'Isl-2'], # 0.67
    ['./data/re/03/state_history.json', 'Isl-3'], # 0.66
    ['./data/re/04/state_history.json', 'Isl-4'], # 0.64
    ['./data/ga-re-a/re3/state_history.json', 'Isl-RE a (1)'], # 0.63
    ['./data/ga-re-a/re6/state_history.json', 'Isl-RE a (2)'], # 0.63
    ['./data/ga-re-b/re3/state_history.json', 'Isl-RE b (1)'], # 0.63
    ['./data/ga-re-b/re6/state_history.json', 'Isl-RE b (2)'], # 0.64
]
# plot_history(f, './report/ga-1-re-b.png', 100, True, 'Island-based RE')

f = [
    ['./data/ga-c-nom/is1/state_history.json', 'Isl-1'], # 0.65
    ['./data/ga-c-nom/is2/state_history.json', 'Isl-2'], # 0.63
    ['./data/ga-c-nom/is3/state_history.json', 'Isl-3'], # 0.67
    ['./data/ga-c-nom/is4/state_history.json', 'Isl-4'], # 0.64
]
# plot_history(f, './report/ga-1-re-b.png', 100, True, 'Island-based GA-2')

f = [
    ['./data/ga-re-b/is1/history.json', 'Isl-GA-1 (1)'], # 0.63
    ['./data/ga-re-b/is2/history.json', 'Isl-GA-1 (2)'], # 0.63
    ['./data/ga-re-b/re3/state_history.json', 'Isl-RE (1)'], # 0.63
    ['./data/ga-re-b/is4/history.json', 'Isl-GA-1 (3)'], # 0.64
    ['./data/ga-re-b/is5/history.json', 'Isl-GA-1 (4)'], # 0.63
    ['./data/ga-re-b/re6/state_history.json', 'Isl-RE (2)'] # 0.64
]
# plot_history(f, './report/ga-1-re-b.png', 100, True, 'Island-based GA-RE')



# GA-3
f = [
    ['./data/ga-isl-a/his_is1.json', 'GA (1)'], # 0.63
    ['./data/ga-isl-a/his_is2.json', 'GA (2)'], # 0.64
    ['./data/ga-isl-a/his_is3.json', 'GA (3)'], # 0.63
]
plot_history(f, './report/res-1-GA.svg', 100, True, '1-tribe GA')

# RE-3
f = [
    ['./data/re/03/state_history.json', 'RE (1)'], # 0.66
    ['./data/re/04/state_history.json', 'RE (2)'], # 0.64
    ['./data/ga-re-a/re3/state_history.json', 'RE (3)'], # 0.63
]
plot_history(f, './report/res-1-RE.svg', 100, True, '1-tribe RE')

# RE-GA-3
f = [
    ['./data/ga-isl-b/his_is2.json', 'GA (1)'], # 0.65
    ['./data/ga-c-nom/is3/state_history.json', 'GA (2)'], # 0.67
    ['./data/ga-re-b/re6/state_history.json', 'RE (3)'], # 0.64
]
plot_history(f, './report/res-1-GA-RE.svg', 100, True, '1-tribe GA-RE')

# C-GA33
f = [
    ['./data/ga-c-nom/is1/state_history.json', 'GA (A1)'], # 0.65
    ['./data/ga-c-nom/is2/state_history.json', 'GA (A2)'], # 0.63
    ['./data/ga-c-nom/is4/state_history.json', 'GA (A3)'], # 0.64
    ['./data/ga-re-b/is1/history.json', 'GA (B1)'], # 0.63
    ['./data/ga-re-b/is2/history.json', 'GA (B2)'], # 0.63
    ['./data/ga-re-b/is4/history.json', 'GA (B3)'], # 0.64
]
plot_history(f, './report/res-2-GA.svg', 100, True, '2-tribe GA')

# C-GA-RE33
f = [
    ['./data/ga-isl-b/his_is1.json', 'GA (A1)'], # 0.68
    ['./data/ga-isl-b/his_is3.json', 'GA (A2)'], # 0.68
    ['./data/re/02/state_history.json', 'RE (A3)'], # 0.67
    ['./data/ga-isl-b/his_is4.json', 'GA (B1)'], # 0.68
    ['./data/ga-isl-a/his_is4.json', 'GA (B2)'], # 0.65
    ['./data/re/01/state_history.json', 'RE (B3)'], # 0.66
]
plot_history(f, './report/res-2-GA-RE.svg', 100, True, '2-tribe GA-RE')