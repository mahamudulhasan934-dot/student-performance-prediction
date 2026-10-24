
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Step 1: Create dataset
data = {
    "study_hours": [
        1, 2, 3, 4, 5, 6, 7, 8, 2, 4,
        6, 3, 5, 7, 1, 8, 9, 2, 5, 10
    ],
    "marks": [
        30, 35, 45, 55, 65, 70, 80, 90, 40, 60,
        75, 48, 68, 78, 25, 88, 95, 38, 62, 98
    ]
}

df = pd.DataFrame(data)

# Step 2: Save and reload CSV
df.to_csv("student_data.csv", index=False)
df = pd.read_csv("student_data.csv")

print("Dataset loaded successfully!")
print(df.head())

# Step 3: Select input and target
X = df[["study_hours"]]
y = df["marks"]

# Step 4: Split data: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 5: Train the model
model = LinearRegression()
model.fit(X_train, y_train)

print("\nModel training completed!")

# Step 6: Test the model
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error:", round(mae, 2))
print("R2 Score:", round(r2, 2))

# Step 7: Predict marks for a new student
hours = float(input("\nEnter daily study hours: "))

if hours < 0:
    print("Study hours cannot be negative.")
else:
    new_student = pd.DataFrame({
        "study_hours": [hours]
    })

    prediction = model.predict(new_student)[0]
    prediction = max(0, min(100, prediction))

    print("Predicted marks:", round(prediction, 2))

# Step 8: Draw and save graph
plt.scatter(
    df["study_hours"],
    df["marks"],
    label="Actual marks"
)

plt.plot(
    df["study_hours"],
    model.predict(df[["study_hours"]]),
    label="Regression line"
)

plt.xlabel("Daily Study Hours")
plt.ylabel("Exam Marks")
plt.title("Student Performance Prediction")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig("student_performance.png")
print("Graph saved as student_performance.png")

plt.show()

# Step 9: Basic data analysis

print("\n--- Dataset Analysis ---")
print("Total students:", len(df))
print("Average study hours:", round(df["study_hours"].mean(), 2))
print("Average marks:", round(df["marks"].mean(), 2))
print("Highest marks:", df["marks"].max())
print("Lowest marks:", df["marks"].min())

print(
    "Correlation between study hours and marks:",
    round(df["study_hours"].corr(df["marks"]), 2)
)
