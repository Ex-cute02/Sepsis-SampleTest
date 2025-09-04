# Dataset Information

## Large Dataset File Excluded

The main dataset file `Dataset.csv` (146.69 MB) has been excluded from the Git repository due to GitHub's 100 MB file size limit.

### To use the system:

1. **For development/testing**: The system includes a smaller sample dataset `M/sepsis_sample_dataset.csv` that can be used for testing purposes.

2. **For full dataset**: If you need the complete dataset:
   - Download it separately from the original source
   - Place it in the root directory as `Dataset.csv`
   - Or use Git LFS (Large File Storage) for version control of large files

### Alternative Solutions:

- **Git LFS**: For future large files, consider using Git LFS
- **Cloud Storage**: Store large datasets in cloud storage and reference them
- **Data Pipeline**: Use data loading scripts that download datasets when needed

The system will work with the sample dataset for demonstration purposes.