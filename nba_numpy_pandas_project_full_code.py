# ==========================================
# NBA PLAYER STATISTICS ANALYSIS PROJECT
# NumPy + Pandas + Matplotlib Project
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. DATA LOADING
# ==========================================

df = pd.read_csv("nba_stats.csv")

print("DATA SUCCESSFULLY LOADED")
print()

# ==========================================
# 2. GENERAL DATA INFORMATION
# ==========================================

print("FIRST 5 ROWS")
print(df.head())

print("\nLAST 5 ROWS")
print(df.tail())

print("\nDATASET SHAPE")
print(df.shape)

print("\nCOLUMN NAMES")
print(df.columns)

print("\nDATA TYPES")
print(df.dtypes)

print("\nGENERAL INFORMATION")
print(df.info())

print("\nSTATISTICAL SUMMARY")
print(df.describe())

# ==========================================
# 3. MISSING VALUE ANALYSIS
# ==========================================

print("\nMISSING VALUES")
print(df.isnull().sum())

# Fill missing numeric values with mean
numeric_columns = df.select_dtypes(include=np.number).columns

for col in numeric_columns:
    df[col] = df[col].fillna(df[col].mean())

# Remove duplicated rows
df = df.drop_duplicates()

print("\nNEW SHAPE AFTER CLEANING")
print(df.shape)

# ==========================================
# 4. BASIC NUMPY OPERATIONS
# ==========================================

points = np.array(df["PTS"])
assists = np.array(df["AST"])
rebounds = np.array(df["REB"])

print("\nNUMPY OPERATIONS")

print("Average Points:", np.mean(points))
print("Median Points:", np.median(points))
print("Maximum Points:", np.max(points))
print("Minimum Points:", np.min(points))
print("Standard Deviation:", np.std(points))

print("90th Percentile:", np.percentile(points, 90))

# ==========================================
# 5. TOP PLAYERS ANALYSIS
# ==========================================

print("\nTOP 10 PLAYERS BY POINTS")

top_scorers = df.nlargest(10, "PTS")

print(top_scorers[["Player", "PTS"]])

print("\nTOP 10 ASSIST PLAYERS")

top_assists = df.nlargest(10, "AST")

print(top_assists[["Player", "AST"]])

print("\nTOP 10 REBOUND PLAYERS")

top_rebounds = df.nlargest(10, "REB")

print(top_rebounds[["Player", "REB"]])

# ==========================================
# 6. FILTERING OPERATIONS
# ==========================================

print("\nPLAYERS UNDER AGE 25")

young_players = df[df["Age"] < 25]

print(young_players[["Player", "Age", "PTS"]])

print("\nPLAYERS WITH 20+ POINTS")

elite_scorers = df[df["PTS"] >= 20]

print(elite_scorers[["Player", "PTS"]])

print("\nLAKERS PLAYERS")

lakers_players = df[df["Team"] == "LAL"]

print(lakers_players[["Player", "Team", "PTS"]])

# ==========================================
# 7. GROUPBY ANALYSIS
# ==========================================

print("\nTEAM BASED POINT AVERAGES")

team_points = df.groupby("Team")["PTS"].mean()

print(team_points)

print("\nPOSITION BASED ASSIST AVERAGES")

position_assists = df.groupby("Position")["AST"].mean()

print(position_assists)

print("\nMULTIPLE AGGREGATION")

team_stats = df.groupby("Team").agg({
    "PTS": "mean",
    "AST": "mean",
    "REB": "mean"
})

print(team_stats)

# ==========================================
# 8. NEW COLUMN CREATION
# ==========================================

df["Performance_Score"] = (
    df["PTS"] * 0.5 +
    df["AST"] * 0.3 +
    df["REB"] * 0.2
)

print("\nPERFORMANCE SCORES")

print(df[["Player", "Performance_Score"]].head())

top_performance = df.nlargest(15, "Performance_Score")

print("\nTOP PERFORMANCE PLAYERS")

print(top_performance[["Player", "Performance_Score"]])

# ==========================================
# 9. SORTING OPERATIONS
# ==========================================

print("\nSORT BY POINTS")

sorted_points = df.sort_values(by="PTS", ascending=False)

print(sorted_points[["Player", "PTS"]].head(10))

# ==========================================
# 10. CORRELATION ANALYSIS
# ==========================================

print("\nCORRELATION MATRIX")

correlation_matrix = df[["PTS", "AST", "REB"]].corr()

print(correlation_matrix)

# ==========================================
# 11. Z-SCORE ANALYSIS
# ==========================================

df["PTS_ZSCORE"] = (
    (df["PTS"] - df["PTS"].mean()) /
    df["PTS"].std()
)

print("\nZ SCORE VALUES")

print(df[["Player", "PTS_ZSCORE"]].head())

# ==========================================
# 12. NORMALIZATION
# ==========================================

df["Normalized_PTS"] = (
    (df["PTS"] - df["PTS"].min()) /
    (df["PTS"].max() - df["PTS"].min())
)

print("\nNORMALIZED POINTS")

print(df[["Player", "Normalized_PTS"]].head())

# ==========================================
# 13. RANDOM PLAYER SELECTION
# ==========================================

random_player = df.sample(1)

print("\nRANDOM PLAYER")

print(random_player[["Player", "PTS", "AST", "REB"]])

# ==========================================
# 14. VISUALIZATION SECTION
# ==========================================

# Histogram
plt.figure(figsize=(8,5))
plt.hist(df["Age"], bins=15)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

# Bar Chart
top10 = df.nlargest(10, "PTS")

plt.figure(figsize=(10,5))
plt.bar(top10["Player"], top10["PTS"])
plt.title("Top 10 Scorers")
plt.xticks(rotation=45)
plt.ylabel("Points")
plt.show()

# Scatter Plot
plt.figure(figsize=(8,5))
plt.scatter(df["AST"], df["PTS"])
plt.title("Assists vs Points")
plt.xlabel("Assists")
plt.ylabel("Points")
plt.show()

# Rebounds Chart
top_reb = df.nlargest(10, "REB")

plt.figure(figsize=(10,5))
plt.bar(top_reb["Player"], top_reb["REB"])
plt.title("Top Rebound Players")
plt.xticks(rotation=45)
plt.ylabel("Rebounds")
plt.show()

# Performance Score Plot
top_perf = df.nlargest(10, "Performance_Score")

plt.figure(figsize=(10,5))
plt.plot(top_perf["Player"], top_perf["Performance_Score"])
plt.title("Performance Scores")
plt.xticks(rotation=45)
plt.ylabel("Score")
plt.show()

# ==========================================
# 15. EXPORT CLEANED DATA
# ==========================================

df.to_csv("cleaned_nba_stats.csv", index=False)

print("\nCLEANED DATA EXPORTED")

# ==========================================
# 16. FINAL STATISTICS
# ==========================================

print("\nFINAL STATISTICS")

print("Average Age:", df["Age"].mean())
print("Average Points:", df["PTS"].mean())
print("Average Assists:", df["AST"].mean())
print("Average Rebounds:", df["REB"].mean())

print("\nHIGHEST SCORER")

highest_scorer = df.loc[df["PTS"].idxmax()]

print(highest_scorer["Player"])
print(highest_scorer["PTS"])

print("\nBEST PERFORMANCE PLAYER")

best_player = df.loc[df["Performance_Score"].idxmax()]

print(best_player["Player"])
print(best_player["Performance_Score"])

# ==========================================
# 17. USER SEARCH SYSTEM
# ==========================================

player_name = input("\nEnter Player Name: ")

searched_player = df[
    df["Player"].str.contains(player_name, case=False)
]

print(searched_player)

# ==========================================
# 18. TEAM FILTER SYSTEM
# ==========================================

team_name = input("\nEnter Team Name: ")

team_data = df[df["Team"] == team_name]

print(team_data)

# ==========================================
# 19. ADVANCED ANALYSIS
# ==========================================

df["Efficiency"] = (
    df["PTS"] +
    df["REB"] +
    df["AST"]
)

top_efficiency = df.nlargest(10, "Efficiency")

print("\nTOP EFFICIENCY PLAYERS")

print(top_efficiency[[
    "Player",
    "Efficiency"
]])

# ==========================================
# 20. PROJECT FINISHED
# ==========================================

print("\nPROJECT COMPLETED SUCCESSFULLY")