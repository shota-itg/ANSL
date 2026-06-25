import statistics

max_load_detail = [2500, 1650, 4800, 2800, 3000, 2850, 2850, 5250]

max_ex = max(max_load_detail)
min_ex = min(max_load_detail)
average_ex = statistics.mean(max_load_detail)
std_dev_ex = statistics.stdev(max_load_detail)
std_err_ex = std_dev_ex / (len(max_load_detail) ** 0.5)
median_ex = statistics.median(max_load_detail)

print(f'Max: {max_ex}, Min: {min_ex}, Average: {average_ex}, SD: {std_dev_ex}, SE: {std_err_ex}, Median: {median_ex}')