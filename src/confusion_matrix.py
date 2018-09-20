# TP TN FP FN

import numpy as np
x = np.array([1, 2, 3, 4, 5],
            [6,7,8,9,10],)

y = np.array([0,1,0,1,0],
            [1,0,1,0,1],)

tp, tn, fp, fn = 0, 0, 0, 0
acc = (tp + tn)/(tp + tn + fp +fn)

recall = tp/(tp+fn)
tnr = tn/(tn+fp)
fnr = fp/(tn+fp)
tpr = fn/(tp+fn)
precision = tp/(tp+fp)

