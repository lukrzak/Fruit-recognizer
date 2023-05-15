import random

import convolutional
import time
import constraints
from matplotlib import pyplot as plt
# rare_set - many classes with few images ~ 100s (conv/5e/64x64).
# small_set - few classes with many images ~ 70s (conv/5e/64x64).
# great_set - many classes with many images




if __name__ == "__main__":
    start = time.time()

    accuracy_conv = convolutional.run(constraints.DATA_PATH, constraints.EPOCHS, (64, 64))
    end = time.time()
    print("Time: " + str(end - start) + " Accuracy: " + str(accuracy_conv))

