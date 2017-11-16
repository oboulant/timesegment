# Libraries
import pandas as pd
import numpy as np

# Own
from timesegment import Partition_tree

# Debug
import matplotlib.pyplot as plt
import time
import cProfile, pstats, io

if __name__ == '__main__':

    data = pd.read_csv('data_sample.csv')
    data = data.iloc[::-1]

    # Start profiling
    # pr = cProfile.Profile()
    # pr.enable()

    # Time
    start_time = time.time()

    decalage = 0

    # mon_obj = Partition_tree(np.array(data['value'])[data.shape[0] - 256-decalage:-decalage],
    #                              -1,
    #                              10,
    #                              0.0,
    #                             1)

    mon_obj = Partition_tree(np.array(data['value'])[data.shape[0] - 256:],
                             -1,
                             10,
                             0.0,
                             1)



    # Time
    start_time = time.time()

    # Start profiling
    pr = cProfile.Profile()
    pr.enable()

    res = mon_obj.split()

    # End profiling
    pr.disable()
    s = io.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())

    # End time
    print("--- %s seconds --- after splitting" % (time.time() - start_time))

    mon_obj.weakest_link_pruning()
    # End time
    print("--- %s seconds --- after pruning" % (time.time() - start_time))
    # print(mon_obj.get_current_nb_segment())
    # durations = mon_obj.get_durations()
    # print(durations)

    preds = mon_obj.get_predictions()

    print("--- %s seconds --- after get_prediction" % (time.time() - start_time))

    durations = mon_obj.get_durations()
    print(durations)

    print("--- %s seconds --- after get_durations" % (time.time() - start_time))


    # plt.plot(np.arange(np.array(data['date'])[data.shape[0] - 256-decalage:-decalage].shape[0]),
    #           np.array(data['value'])[data.shape[0] - 256-decalage:-decalage], 'k',
    #          np.arange(np.array(data['date'])[data.shape[0] - 256-decalage:-decalage].shape[0]),
    #          preds, 'ro')
    plt.plot(np.arange(np.array(data['date'])[data.shape[0] - 256:].shape[0]),
             np.array(data['value'])[data.shape[0] - 256:], 'k',
             np.arange(np.array(data['date'])[data.shape[0] - 256:].shape[0]),
             preds, 'ro')
    plt.show()


