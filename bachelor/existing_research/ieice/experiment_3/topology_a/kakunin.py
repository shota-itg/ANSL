import statistics
import math

a = [1500, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3200, 5600]

max = max(a)
min = min(a)
average = statistics.mean(a)
std_dev = statistics.stdev(a)
# std_err = std_dev / (len(a) ** 0.5)
std_err = std_dev / math.sqrt(len(a))
median = statistics.median(a)

print(f'len(a): {len(a)}, Max: {max}, Min: {min}, Average: {average}, SD: {std_dev}, SE: {std_err}, Median: {median}')
print()