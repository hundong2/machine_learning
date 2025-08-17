import pandas as pd
import random
from pathlib import Path

base_dir = Path(__file__).parent
train_path = base_dir / 'train_used_car_sample.csv'
test_path = base_dir / 'test_used_car_sample.csv'

brand_choices = ['Toyota', 'Ford', 'Honda', 'Chevrolet']
brand_weights = [0.4, 0.3, 0.2, 0.1]
model_choices = ['Sedan', 'SUV', 'Truck']
model_weights = [0.5, 0.3, 0.2]
maint_choices = ['없음', '보통', '정기적']
maint_weights = [0.4, 0.3, 0.3]

brand_base = {'Toyota': 23000, 'Ford': 20000, 'Honda': 19500, 'Chevrolet': 18500}
model_adj = {'Sedan': 0, 'SUV': 2000, 'Truck': 1000}
maint_adj = {'없음': 0, '보통': 500, '정기적': 1500}

random.seed(123)

# Generate 100 train rows
train_rows = []
for _ in range(100):
    b = random.choices(brand_choices, weights=brand_weights, k=1)[0]
    m = random.choices(model_choices, weights=model_weights, k=1)[0]
    g = random.choices(maint_choices, weights=maint_weights, k=1)[0]
    mileage = round(random.uniform(30000, 90000), 1)
    price = brand_base[b] + model_adj[m] + maint_adj[g] - (mileage * 0.08) + random.uniform(-800, 800)
    price = round(max(price, 5000.0), 1)
    train_rows.append({'Brand': b, 'Model': m, '정비 이력등급': g, '주행거리': mileage, '중고차가격': price})

synthetic_train = pd.DataFrame(train_rows)

# Generate 100 test rows
test_rows = []
for _ in range(100):
    b = random.choices(brand_choices, weights=brand_weights, k=1)[0]
    m = random.choices(model_choices, weights=model_weights, k=1)[0]
    g = random.choices(maint_choices, weights=maint_weights, k=1)[0]
    mileage = round(random.uniform(30000, 90000), 1)
    test_rows.append({'Brand': b, 'Model': m, '정비 이력등급': g, '주행거리': mileage})

synthetic_test = pd.DataFrame(test_rows)

# Append to CSVs
existing_train = pd.read_csv(train_path)
existing_test = pd.read_csv(test_path)

updated_train = pd.concat([existing_train, synthetic_train], ignore_index=True)
updated_test = pd.concat([existing_test, synthetic_test], ignore_index=True)

updated_train.to_csv(train_path, index=False)
updated_test.to_csv(test_path, index=False)

print(f"train rows: {len(existing_train)} -> {len(updated_train)}")
print(f"test rows: {len(existing_test)} -> {len(updated_test)}")
