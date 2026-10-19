import pandas as pd
import time
from concurrent.futures import ThreadPoolExecutor

df = pd.read_csv("cleaned_health_data.csv")

df["Cases"] = pd.to_numeric(df["Cases"], errors="coerce")
df["year"] = pd.to_numeric(df["year"], errors="coerce")
df = df.dropna(subset=["Cases"])

# Three independent analysis tasks
def disease_analysis():
    return df.groupby("Disease")["Cases"].sum()

def state_analysis():
    return df.groupby("state_ut")["Cases"].sum()

def yearly_analysis():
    return df.groupby("year")["Cases"].sum()

tasks = [
    disease_analysis,
    state_analysis,
    yearly_analysis
]

# Sequential execution
start = time.perf_counter()

for task in tasks:
    result = task()

sequential_time = time.perf_counter() - start

# Multithreaded execution
start = time.perf_counter()

with ThreadPoolExecutor(max_workers=3) as executor:
    results = list(executor.map(lambda task: task(), tasks))

parallel_time = time.perf_counter() - start

print("PUBLIC HEALTH MULTITHREADING ANALYSIS")
print("-------------------------------------")
print("Disease groups:", len(results[0]))
print("States analyzed:", len(results[1]))
print("Years analyzed:", len(results[2]))
print(f"Sequential time: {sequential_time:.6f} seconds")
print(f"Multithreaded time: {parallel_time:.6f} seconds")

if sequential_time > 0:
    speedup = sequential_time / parallel_time
    print(f"Speedup: {speedup:.2f}x")

print("Analysis completed!")
