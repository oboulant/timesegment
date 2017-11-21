# Automatiocally segment a Timeseries into segment

## Installation

```
> pip install git+https://github.com/oboulant/timesegment.git
```

## Example

```python
import pandas as pd
import numpy as np

from timesegment import Partition_tree

import matplotlib.pyplot as plt

# Read data
data = pd.read_csv('data_sample.csv')
# Invert time axis (specific for this data)
data = data.iloc[::-1]

# Segment the 256 most recent points
my_obj = Partition_tree(np.array(data['value'])[data.shape[0] - 256:], # data as numpy array
                             -1,  # Max depth of the partitionning tree
                             1,   # Early Stop
                             30,  # The number of segment desired after pruning
                             0.0, # A Complexity parameter
                             1)   # Tau : Minimum number of observations within a segment

# Build the partition tree
res = mon_obj.split()
# Tree pruning
mon_obj.weakest_link_pruning()
# Get predictions
preds = mon_obj.get_predictions()
# Get segments durations
durations = mon_obj.get_durations()
print(durations)

# Plot raw data alongside with prediction
plt.plot(np.arange(np.array(data['date'])[data.shape[0] - 256:].shape[0]),
             np.array(data['value'])[data.shape[0] - 256:], 'k',
             np.arange(np.array(data['date'])[data.shape[0] - 256:].shape[0]),
             preds, 'ro')
plt.show()
```

## Object Partition_tree parameters

* `signal` : The timeseries to be segmented. It should be a numpy array and of shape = [n_samples]
* `max_depth` : The maximum depth of the partitioning tree. If -1, then no depth constraint exists on the tree
* `early_stop` : Early Stop. If 1, then stop splitting a node if no MSE improvment is found. Otherwise, the best split is performed, even if it induces a MSE increase.
* `nb_segments` : The number of segments desired when performing the Weakest Link Pruning
* `delta_complexity` : A complexity parameter. Only perform the best split if `np.abs(MSE_CurrentNode - min(MSE_LeftChild + MSE_RightChild)) / MSE_CurrentNode <= delta_complexity` Which in human language reads as : "only perform the best split if it decreases the MSE by more than delta_complexity percentage"
* `tau` : The minimum number of observation within a segment. If the current segment has less than `2*tau` observation, we do not split. Otherwise, we split in two segments, both of which of duration greater than `tau`

## Still to be done

* Change the call to `np.append()` in 
    * `Partition_node.get_predictions()`
    * `Partition_node.get_durations()`

