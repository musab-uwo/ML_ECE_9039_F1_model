import fastf1
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

fastf1.Cache.enable_cache("f1_cache")


session_2024 = fastf1.get_session(2024, 3, "R")
session_2024.load()

laps_2024 = session_2024.laps[["Driver", "LapTime"]].copy()
laps_2024.dropna(subset=["LapTime"], inplace=True)
laps_2024["LapTime (s)"] = laps_2024["LapTime"].dt.total_seconds()


qualifying_2025 = pd.DataFrame({
    "Driver": ["Lando Norris", "Oscar Piastri", "Max Verstappen", "George Russell", "Yuki Tsunoda",
               "Alexander Albon", "Charles Leclerc", "Lewis Hamilton", "Pierre Gasly", "Carlos Sainz", "Fernando Alonso", "Lance Stroll"],
    "QualifyingTime (s)": [75.096, 75.180, 75.481, 75.546, 75.670,
                           75.737, 75.755, 75.973, 75.980, 76.062, 76.4, 76.5]
})


driver_mapping = {
    "Lando Norris": "NOR", "Oscar Piastri": "PIA", "Max Verstappen": "VER", "George Russell": "RUS",
    "Yuki Tsunoda": "TSU", "Alexander Albon": "ALB", "Charles Leclerc": "LEC", "Lewis Hamilton": "HAM",
    "Pierre Gasly": "GAS", "Carlos Sainz": "SAI", "Lance Stroll": "STR", "Fernando Alonso": "ALO"
}

qualifying_2025["DriverCode"] = qualifying_2025["Driver"].map(driver_mapping)


merged_data = qualifying_2025.merge(laps_2024, left_on="DriverCode", right_on="Driver")


X = merged_data[["QualifyingTime (s)"]]
y = merged_data["LapTime (s)"]

if X.shape[0] == 0:
    raise ValueError("Dataset is empty after preprocessing. Check data sources!")


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=39)
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=39)
model.fit(X_train, y_train)


predicted_lap_times = model.predict(qualifying_2025[["QualifyingTime (s)"]])
qualifying_2025["PredictedRaceTime (s)"] = predicted_lap_times


qualifying_2025 = qualifying_2025.sort_values(by="PredictedRaceTime (s)")


print("\n🏁 Predicted 2025 Chinese GP Winner 🏁\n")
print(qualifying_2025[["Driver", "PredictedRaceTime (s)"]])


y_pred = model.predict(X_test)
print(f"\n🔍 Model Error (MAE): {mean_absolute_error(y_test, y_pred):.2f} seconds")
