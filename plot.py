from matplotlib.pyplot import *

x = [5, 15, 25, 35, 45]
y1 = [0.742201, 0.782059, 0.798079, 0.801288, 0.796246]
y2 = [0.768548, 0.763095, 0.773253, 0.783941, 0.736917]

plot(x, y2, color='b', marker='o', linestyle='-')
title("Accuracy vs word vector dimension for RNN2.")
xlabel("wvecdim")
ylabel("Accuracy")
#ylim(ymin=0, ymax=max(1.1*max(train_accuracies),3*min(train_accuracies)))
show()
