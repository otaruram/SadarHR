import pandas as pd
import numpy as np

def generate_dummy_data():
    np.random.seed(42)
    n_samples = 500

    data = {
        'EmployeeID': range(1, n_samples + 1),
        'Age': np.random.randint(22, 60, size=n_samples),
        'JobSatisfaction': np.random.choice(['Low', 'Medium', 'High', 'Very High'], size=n_samples),
        'OverTime': np.random.choice(['Yes', 'No'], size=n_samples, p=[0.3, 0.7]),
        'Department': np.random.choice(['Sales', 'Research & Development', 'Human Resources'], size=n_samples),
        'MonthlyIncome': np.random.randint(2000, 20000, size=n_samples),
        'YearsAtCompany': np.random.randint(0, 40, size=n_samples),
        'Attrition': np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2]) # 0 = No, 1 = Yes
    }

    df = pd.DataFrame(data)
    df.to_csv('train.csv', index=False)
    print("Generated dummy train.csv")

if __name__ == '__main__':
    generate_dummy_data()
