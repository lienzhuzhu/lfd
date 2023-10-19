import numpy as np
import matplotlib.pyplot as plt

COINS = 1000
FLIPS = 10
TRIALS = 100000

first_coin_heads = np.zeros(TRIALS)
random_coin_heads = np.zeros(TRIALS)
min_heads_coin_heads = np.zeros(TRIALS)


for i in range(TRIALS):
    # Simulate flips for all coins
    flips = np.random.randint(2, size=(COINS, FLIPS))
    heads_counts = flips.sum(axis=1)
    
    # Record data for the first coin
    first_coin_heads[i] = heads_counts[0]
    
    # Record data for a randomly chosen coin
    random_coin_index = np.random.choice(COINS)
    random_coin_heads[i] = heads_counts[random_coin_index]
    
    # Record data for the coin with the minimum frequency of heads
    min_heads_coin_index = np.argmin(heads_counts)
    min_heads_coin_heads[i] = heads_counts[min_heads_coin_index]


# Plotting the results
plt.figure(figsize=(15,5))

plt.subplot(1, 3, 1)
plt.hist(first_coin_heads, bins=range(12), alpha=0.7, edgecolor='black')
plt.title("First Coin")
plt.xlabel("Number of Heads")
plt.ylabel("Frequency")

plt.subplot(1, 3, 2)
plt.hist(random_coin_heads, bins=range(12), alpha=0.7, edgecolor='black')
plt.title("Randomly Chosen Coin")
plt.xlabel("Number of Heads")

plt.subplot(1, 3, 3)
plt.hist(min_heads_coin_heads, bins=range(12), alpha=0.7, edgecolor='black')
plt.title("Coin with Minimum Heads")
plt.xlabel("Number of Heads")

plt.tight_layout()
plt.show()
