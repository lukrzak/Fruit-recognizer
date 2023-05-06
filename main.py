import convolutional
import time

# rare_set - many classes with few images ~ 100s (conv/5e/64x64).
# small_set - few classes with many images ~ 70s (conv/5e/64x64).
# great_set - many classes with many images

DATA_SET = 'rare_set'
DATA_PATH = 'data/' + DATA_SET


if __name__ == "__main__":
    start = time.time()
    accuracy_conv = convolutional.run(DATA_PATH, 5, (64, 64))
    end = time.time()
    print("Time: " + str(end - start) + " Accuracy: " + str(accuracy_conv))

