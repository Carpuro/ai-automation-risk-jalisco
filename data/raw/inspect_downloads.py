import pandas as pd, warnings
warnings.filterwarnings('ignore')

aioe = pd.read_excel('AIOE_DataAppendix.xlsx', sheet_name=None)
print('=== AIOE_DataAppendix.xlsx ===')
for sheet, df in aioe.items():
    print(f'  [{sheet}] {df.shape}')
    print(f'  cols: {list(df.columns[:8])}')
    print(df.head(2).to_string())
    print()

lm = pd.read_excel('Language_Modeling_AIOE_AIIE.xlsx', sheet_name=None)
print('=== Language_Modeling_AIOE_AIIE.xlsx ===')
for sheet, df in lm.items():
    print(f'  [{sheet}] {df.shape}')
    print(f'  cols: {list(df.columns[:8])}')
    print(df.head(2).to_string())
    print()

job = pd.read_csv('anthropic_job_exposure.csv')
print(f'=== anthropic_job_exposure.csv === {job.shape}')
print(f'cols: {list(job.columns)}')
print(job.head(5).to_string())

print()
task = pd.read_csv('anthropic_task_penetration.csv')
print(f'=== anthropic_task_penetration.csv === {task.shape}')
print(f'cols: {list(task.columns)}')
print(task.head(5).to_string())
