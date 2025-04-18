import pandas as pd
from tqdm import tqdm
from generate_password import generate_strong_password
# Step 1: Load data from existing CSV file
df = pd.read_csv("data.csv")

# Step 2: Filter class 1 and class 2 samples
class1_df = df[df['strength'] == '0'].sample(n=2000, random_state=42)
class2_df = df[df['strength'] == '1'].sample(n=2000, random_state=42)

# Step 3: Generate class 3 passwords (strong) — only store password + strength
generated_passwords = []
for _ in tqdm(range(2000), desc="Generating strong passwords"):
    pwd = generate_strong_password()
    generated_passwords.append({
        'password': pwd,
        'strength': '2'
    })

class3_df = pd.DataFrame(generated_passwords)

# Step 4: Combine all and save
final_df = pd.concat([class1_df[['password', 'strength']], 
                      class2_df[['password', 'strength']], 
                      class3_df], ignore_index=True)

final_df.to_csv("balanced_password_dataset.csv", index=False)

print("✅ Saved balanced dataset with 6,000 samples to 'balanced_password_dataset.csv'")
