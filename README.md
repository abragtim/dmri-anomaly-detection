## Preprocessing

1. run `python3 clear_raw_data.py` (use --dry-run to test)
2. run `python3 extract_hcp_data.py --cleanup --output-dir ./extracted_hcp_data` (use --dry-run to test)
3. run `python3 generate_metrics.py --output-dir ./extracted_mean_diffusivity_hcp_data` (use --dry-run to test)
4. run `python3 convert_to_yucca_format.py --test-set-amount X` (X is the number of test samples, e.g. 10, use --dry-run to test)
5. setup [environment variables for yucca](https://github.com/Sllambias/yucca/blob/main/yucca/documentation/guides/environment_variables.md), then run `cp .env /Users/timur/anaconda3/envs/diffusion-segmentation/lib/python3.12/site-packages/yucca/.env`
6. run `yucca_preprocess -t Task001_DMRIHCP152 -pr UnsupervisedPreprocessor` (remove metadata files at the first run, then run again; use --dry-run to test)
7. run `yucca_train -t Task001_DMRIHCP152 -m TinyUNet -d 2D --loss MSE`
8. run `PYTORCH_ENABLE_MPS_FALLBACK=1 yucca_inference -s Task001_DMRIHCP152 -t Task001_DMRIHCP152 -m TinyUNet -d 2D`

## Q&A

### How to handle macOS metadata files (._ files)?

**Problem**: macOS creates `._ files` (AppleDouble files) when copying files to non-HFS+ file systems (external drives, network shares, etc.). These files contain extended attributes and metadata.

**Solution 1: Prevent creation of ._ files (recommended)**

todo: maybe does not work

```bash
# Run once to disable creation of metadata files permanently
./disable_metadata_files.sh
```
This script:
- Sets `COPYFILE_DISABLE=1` environment variable permanently
- Disables .DS_Store files on network and USB volumes  
- Applies to your shell profile (.zshrc/.bashrc) for future sessions

**Solution 2: Remove existing ._ files**
```bash
# Preview what files would be removed (safe to run)
python3 remove_metadata_files.py --dry-run

# Actually remove all ._ files from current directory and subdirectories
python3 remove_metadata_files.py
```

**Recommended workflow:**
1. First run `./disable_metadata_files.sh` to prevent future ._ files
2. Then run `python3 remove_metadata_files.py --dry-run` to preview cleanup
3. Finally run `python3 remove_metadata_files.py` to clean existing files
