import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

data = {
    "Name": ["Ali", "Sara", "John", "Ayesha", "Bilal", "Nida"],
    "Age": [20, 19, np.nan, 21, 22, 20],
    "Gender": ["Male", "Female", "Male", "F", "Male", "Female"],
    "Marks": [85, np.nan, 45, 92, -10, 73],
    "Grade": ["A", "B", "C", "A", "F", "B"],
    "City": ["Karachi", "Lahore", "lahore", "Islamabad", "Karachi", np.nan]
}

df = pd.DataFrame(data)
print("Raw Data:\n", df)

df['Age'] = (df['Age'].fillna(df['Age'].mean()))

df['Marks'] = df['Marks'].fillna(df['Marks'].median())

df['City'] = df['City'].fillna("Unknown")

df['Gender'] = df['Gender'].replace({"F":"Female","M":"Male"})

df['City'] = df["City"].str.capitalize()

df['Marks'] = df['Marks'].apply(lambda x: 0 if x < 0 else x)

scaler = MinMaxScaler()
df[['Age', 'Marks']] = scaler.fit_transform(df[['Age', 'Marks']])

print("Raw Data:\n", df)