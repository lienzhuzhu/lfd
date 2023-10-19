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
#plt.figure(figsize=(15,5))
#
#plt.subplot(1, 3, 1)
#plt.hist(first_coin_heads, bins=range(12), alpha=0.7, edgecolor='black')
#plt.title("First Coin")
#plt.xlabel("Number of Heads")
#plt.ylabel("Frequency")
#
#plt.subplot(1, 3, 2)
#plt.hist(random_coin_heads, bins=range(12), alpha=0.7, edgecolor='black')
#plt.title("Randomly Chosen Coin")
#plt.xlabel("Number of Heads")
#
#plt.subplot(1, 3, 3)
#plt.hist(min_heads_coin_heads, bins=range(12), alpha=0.7, edgecolor='black')
#plt.title("Coin with Minimum Heads")
#plt.xlabel("Number of Heads")
#
#plt.tight_layout()
#plt.show()



def hoeffding_bound(epsilon, N):
    return 2 * np.exp(-2 * epsilon**2 * N)

epsilons = np.arange(0, 0.5, 0.01)

first_coin_probs = [np.mean(np.abs(first_coin_heads/FLIPS - 0.5) > eps) for eps in epsilons]
random_coin_probs = [np.mean(np.abs(random_coin_heads/FLIPS - 0.5) > eps) for eps in epsilons]
min_heads_coin_probs = [np.mean(np.abs(min_heads_coin_heads/FLIPS - 0.5) > eps) for eps in epsilons]

hoeffding_probs = hoeffding_bound(epsilons, FLIPS)




# Plotting the results
plt.figure(figsize=(15, 15))

# Histograms
plt.subplot(3, 3, 1)
plt.hist(first_coin_heads, bins=range(12), alpha=0.7, edgecolor='black')
plt.title("First Coin Histogram")
plt.xlabel("Number of Heads")
plt.ylabel("Frequency")

plt.subplot(3, 3, 2)
plt.hist(random_coin_heads, bins=range(12), alpha=0.7, edgecolor='black')
plt.title("Randomly Chosen Coin Histogram")
plt.xlabel("Number of Heads")

plt.subplot(3, 3, 3)
plt.hist(min_heads_coin_heads, bins=range(12), alpha=0.7, edgecolor='black')
plt.title("Coin with Minimum Heads Histogram")
plt.xlabel("Number of Heads")

# Hoeffding Plots
plt.subplot(3, 3, 4)
plt.plot(epsilons, first_coin_probs, label="Experimental")
plt.plot(epsilons, hoeffding_probs, label="Hoeffding Bound", linestyle='--')
plt.title("First Coin")
plt.xlabel("Epsilon (ε)")
plt.ylabel("Probability Measure")
plt.legend()

plt.subplot(3, 3, 5)
plt.plot(epsilons, random_coin_probs, label="Experimental")
plt.plot(epsilons, hoeffding_probs, label="Hoeffding Bound", linestyle='--')
plt.title("Randomly Chosen Coin")
plt.xlabel("Epsilon (ε)")

plt.subplot(3, 3, 6)
plt.plot(epsilons, min_heads_coin_probs, label="Experimental")
plt.plot(epsilons, hoeffding_probs, label="Hoeffding Bound", linestyle='--')
plt.title("Coin with Minimum Heads")
plt.xlabel("Epsilon (ε)")

plt.tight_layout()
plt.show()
