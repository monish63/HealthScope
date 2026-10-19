
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Load cleaned dataset
df = pd.read_csv("cleaned_health_data.csv")

# Prepare required columns
df["Cases"] = pd.to_numeric(df["Cases"], errors="coerce")
df["year"] = pd.to_numeric(df["year"], errors="coerce")
df = df.dropna(subset=["Cases"])

# Graph 1: Year-wise cases
yearly = df.dropna(subset=["year"]).groupby("year")["Cases"].sum().sort_index()

plt.figure(figsize=(9, 5))
yearly.plot(kind="line", marker="o")
plt.title("Year-wise Public Health Cases")
plt.xlabel("Year")
plt.ylabel("Total Cases")
plt.grid(True)
plt.tight_layout()
plt.savefig("yearly_cases.png")
plt.close()

# Graph 2: Top 10 states
states = df.groupby("state_ut")["Cases"].sum()
states = states.sort_values(ascending=False).head(10).sort_values()

plt.figure(figsize=(10, 6))
states.plot(kind="barh")
plt.title("Top 10 States by Total Cases")
plt.xlabel("Total Cases")
plt.ylabel("State / Union Territory")
plt.tight_layout()
plt.savefig("state_cases.png")
plt.close()

# Graph 3: Top 10 diseases
diseases = df.groupby("Disease")["Cases"].sum()
diseases = diseases.sort_values(ascending=False).head(10).sort_values()

plt.figure(figsize=(10, 6))
diseases.plot(kind="barh")
plt.title("Top 10 Diseases by Total Cases")
plt.xlabel("Total Cases")
plt.ylabel("Disease")
plt.tight_layout()
plt.savefig("disease_cases.png")
plt.close()

print("All three graphs generated successfully!")
print("1. yearly_cases.png")
print("2. state_cases.png")
print("3. disease_cases.png")
