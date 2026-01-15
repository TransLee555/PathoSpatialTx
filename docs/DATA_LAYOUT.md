# Expected Input Data Layout

The input data for the project is organized into three stages, each with its own folder structure. Detailed below are the expected layouts for each stage:

## Stage 1: Raw Data
```
raw_data/
├── experiment_1/
│   ├── sample1.csv
│   ├── sample2.csv
├── experiment_2/
│   ├── sample1.csv
│   └── sample2.csv
```

## Stage 2: Processed Data
```
processed_data/
├── experiment_1/
│   ├── normalized_sample1.csv
│   ├── normalized_sample2.csv
├── experiment_2/
│   ├── normalized_sample1.csv
│   └── normalized_sample2.csv
```

## Stage 3: Analysis Results
```
analysis_results/
├── experiment_1/
│   ├── results_summary.csv
│   ├── detailed_results/
│   │   ├── analysis1.csv
│   │   └── analysis2.csv
├── experiment_2/
│   ├── results_summary.csv
│   └── detailed_results/
│       ├── analysis1.csv
│       └── analysis2.csv
```

Please ensure that the data adheres to these structures to allow for seamless processing and analysis.