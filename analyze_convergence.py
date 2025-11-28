"""
Scientific convergence analysis based on academic literature.
"""
import pandas as pd
import numpy as np
import sys

# Load evolution data
try:
    df = pd.read_csv('results/evolution_data_20251128_172228.csv')
except:
    print("Error: Could not load evolution data")
    sys.exit(1)

print('=' * 70)
print('PHAN TICH KHOA HOC VE HOI TU (SCIENTIFIC CONVERGENCE ANALYSIS)')
print('=' * 70)
print()

# 1. Convergence Criteria Analysis (De Jong 1975, Eiben & Smith 2003)
print('1. TIEU CHI HOI TU (Convergence Criteria):')
print('   Reference: De Jong (1975), Eiben & Smith (2003)')
print('   Definition: "Population homogeneous + No improvement"')
print('   Quantitative: Coefficient of Variation (CV) < 0.05')
print()

# Calculate coefficient of variation for each generation
df['cv_fitness'] = df['std_fitness'] / abs(df['avg_fitness'])

# Check last 100 generations
last_100 = df.tail(100)
cv_last_100 = last_100['cv_fitness'].mean()

print(f'   CV cua 100 gen cuoi: {cv_last_100:.4f}')
print(f'   Nguong hoi tu ly thuyet: < 0.05')
print(f'   Result: {"DA HOI TU" if cv_last_100 < 0.05 else "CHUA HOI TU"} (CV = {cv_last_100:.4f})')
print()

# 2. Improvement Rate Analysis (Goldberg 1989)
print('2. TOC DO CAI THIEN (Improvement Rate):')
print('   Reference: Goldberg (1989)')
print('   Criterion: Stagnation if improvement < 0.1% over 100 generations')
print()

# Last 200 generations improvement
last_200_start = df.iloc[-200]['best_distance']
last_200_end = df.iloc[-1]['best_distance']
improvement_last_200 = (last_200_start - last_200_end) / last_200_start * 100

print(f'   Cai thien trong 200 gen cuoi: {improvement_last_200:.2f}%')
print(f'   Best distance at gen 800: {last_200_start:.2f}')
print(f'   Best distance at gen 1000: {last_200_end:.2f}')
print(f'   Result: {"STAGNANT" if improvement_last_200 < 0.1 else "VAN CON CAI THIEN"} ({improvement_last_200:.2f}%)')
print()

# 3. Population Diversity Analysis (Srinivas & Patnaik 1994)
print('3. DA DANG QUAN THE (Population Diversity):')
print('   Reference: Srinivas & Patnaik (1994)')
print('   Warning threshold: > 90% diversity drop = premature convergence')
print()

df['diversity_normalized'] = df['diversity'] / df['diversity'].max()

diversity_start = df.head(100)['diversity_normalized'].mean()
diversity_end = df.tail(100)['diversity_normalized'].mean()
diversity_drop = (diversity_start - diversity_end) / diversity_start * 100

print(f'   Diversity dau (gen 0-100): {diversity_start:.4f}')
print(f'   Diversity cuoi (gen 900-1000): {diversity_end:.4f}')
print(f'   Giam: {diversity_drop:.2f}%')
print(f'   Result: {"Diversity tot" if diversity_drop < 90 else "MAT DIVERSITY"} ({diversity_drop:.2f}% drop)')
print()

# 4. Fitness Variance Analysis (Whitley 1989)
print('4. PHUONG SAI FITNESS (Fitness Variance):')
print('   Reference: Whitley (1989)')
print('   Expected: Variance should decrease to < 30% of initial')
print()

variance_start = df.head(100)['std_fitness'].mean()
variance_end = df.tail(100)['std_fitness'].mean()
variance_ratio = variance_end / variance_start

print(f'   Std fitness dau (gen 0-100): {variance_start:.2f}')
print(f'   Std fitness cuoi (gen 900-1000): {variance_end:.2f}')
print(f'   Ty le: {variance_ratio:.4f} (= {variance_ratio*100:.1f}% of initial)')
print(f'   Result: {"ON DINH" if variance_ratio < 0.3 else "KHONG ON DINH"} ({variance_ratio*100:.1f}%)')
print()

# 5. Premature Convergence Detection (Michalewicz 1996)
print('5. PHAT HIEN HOI TU SOM (Premature Convergence):')
print('   Reference: Michalewicz (1996)')
print('   Symptom: No improvement BUT diversity still high = stuck in local optimum')
print()

best_plateau = improvement_last_200 < 1.0
diversity_still_high = diversity_end > 0.3

print(f'   Best plateau (< 1% improvement): {best_plateau}')
print(f'   Diversity still high (> 0.3): {diversity_still_high}')

if best_plateau and diversity_still_high:
    print('   WARNING: Dau hieu hoi tu som - GA stuck o local optimum')
else:
    print('   OK: Khong co dau hieu hoi tu som ro rang')
print()

# 6. Generation Sufficiency Analysis
print('6. DU GENERATIONS CHUA? (Generation Sufficiency):')
print('   Formula: De Jong (1975): G >= 100 * sqrt(n)')
print('   Formula: Potvin & Bengio (1996) VRP: G >= 1500 for n=100')
print()

n_customers = 20
dejong_recommended = int(100 * np.sqrt(n_customers))
vrp_recommended = 1500 if n_customers >= 50 else 1000

print(f'   So customers: {n_customers}')
print(f'   De Jong formula: >= {dejong_recommended} generations')
print(f'   VRP benchmark (Potvin & Bengio): >= {vrp_recommended} generations')
print(f'   Thuc te chay: 1000 generations')
print(f'   Result: {"DU" if 1000 >= max(dejong_recommended, vrp_recommended) else "CHUA DU"}')
print()

# Summary
print('=' * 70)
print('KET LUAN KHOA HOC (SCIENTIFIC CONCLUSIONS):')
print('=' * 70)
print()

converged_math = cv_last_100 < 0.05
still_improving = improvement_last_200 > 0.1
population_stable = variance_ratio < 0.3
diversity_ok = diversity_drop < 90

print(f'1. Hoi tu toan hoc (CV < 0.05): {"YES" if converged_math else "NO"} - CV={cv_last_100:.4f}')
print(f'2. Van dang cai thien: {"YES" if still_improving else "NO"} - {improvement_last_200:.2f}%')
print(f'3. Quan the on dinh (variance < 30%): {"YES" if population_stable else "NO"} - {variance_ratio*100:.1f}%')
print(f'4. Diversity tot (drop < 90%): {"YES" if diversity_ok else "NO"} - {diversity_drop:.1f}%')
print()

# Diagnosis
print('CHAN DOAN (DIAGNOSIS):')
print('-' * 70)

if not converged_math and still_improving:
    print('GA CHUA HOI TU - CAN THEM GENERATIONS')
    print('Evidence:')
    print('  - CV > 0.05: Population chua dong nhat')
    print('  - Van cai thien {:.2f}% trong 200 gen cuoi'.format(improvement_last_200))
    print('  - De Jong formula: Can it nhat {} gen'.format(dejong_recommended))
    print()
    print('RECOMMENDATION: Tang generations len 1500-2000')

elif converged_math and not still_improving:
    print('GA DA HOI TU VE LOCAL OPTIMUM')
    print('Evidence:')
    print('  - CV < 0.05: Population da dong nhat')
    print('  - Khong cai thien nua: {:.2f}% trong 200 gen cuoi'.format(improvement_last_200))
    print()
    print('RECOMMENDATION: Tang population size de mo rong search space')

elif not population_stable:
    print('QUAN THE KHONG ON DINH - CAN TANG POPULATION')
    print('Evidence:')
    print('  - Variance van con {:.1f}% of initial'.format(variance_ratio*100))
    print('  - Goldberg formula: Pop = 1.65 * sqrt(l * k)')
    print('  - Voi n=20, nen dung population >= 150-200')
    print()
    print('RECOMMENDATION: Tang population tu 100 len 200')
else:
    print('TRANG THAI KHONG RO RANG - CAN PHAN TICH THEM')

print()
print('=' * 70)
