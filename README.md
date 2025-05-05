## Usage

- `--classical_gflownet`: Train and evaluate the Classical GFlowNet.
- `--comparison_gflownet`: Train and evaluate the Comparison GFlowNet.
- `--all`: Train and evaluate both Classical and Comparison GFlowNets.

### Examples

1. **Train and Evaluate Only the Classical GFlowNet:**

   ```bash
   python main.py --classical_gflownet
   ```
   
2. **Train and Evaluate Only the Comparison GFlowNet:**
   ```bash
   python main.py --comparison_gflownet
    ```
   
3. **Train and Evaluate Both Models Simultaneously:**
   ```bash
    python main.py --all
   ```
4. **Use Random Baseline:**
   ```bash
    python main.py --random_baseline
   ```
5. **Use COCO dataset:**   
```bash
python main.py --dataset=COCO --all --random_baseline --model_type=convnext --num_iterations=10000 --subset_size=10 --target_classes_coco="person:1,train:2,suitcase:2"
```
