#!/usr/bin/env python3

import os
import json
import numpy as np
import pandas as pd
import time
import gc
import psutil
import matplotlib.pyplot as plt

# Assuming your project structure for these imports
from unidecode import unidecode
# === NOTE: This assumes 'spacymodel.py' is in a folder named 'packages' ===
from packages.spacy_model import SpacyModel 

# === NOTE: These imports will fail unless run in a mock Triton env ===
# === I am assuming you have a way to handle this, as per your file ===
from triton_python_backend_utils import Tensor
from tests.mocks import mockInferenceRequest, Request 
# ^Assuming 'Request' comes from mocks? If not, this is still broken.


# --- Utilities for Memory Profiling and Reporting ---

class MemoryProfiler:
    """Memory monitoring and data collection utilities"""
    def __init__(self):
        self.process = psutil.Process(os.getpid())
    def get_memory_mb(self) -> float:
        return self.process.memory_info().rss / 1024 / 1024
    def get_vocab_stats(self, nlp) -> dict:
        """Get vocabulary and string store statistics"""
        return {
            'vocab_size': len(nlp.vocab),
            'string_store_size': len(nlp.vocab.strings),
        }

def generate_report(results: list, output_path: str):
    """Generates and saves a visual report of memory and store consumption."""
    print("\nGENERATING INVESTIGATION REPORT...")
    fig, axes = plt.subplots(1, 2, figsize=(20, 6)) # Made plot wider
    fig.suptitle('NER Pipeline Memory Investigation', fontsize=16, fontweight='bold')

    # --- Plot 1: Consumption Over Time ---
    ax1 = axes[0]
    ax1.set_title('Consumption Over Batches')
    ax1.set_xlabel('Batch Number')
    ax1.set_ylabel('Total Memory Usage (MB)', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True, axis='y', linestyle='--', alpha=0.6)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Vocab / String Store Size', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    # Define colors for 3 tests
    mem_colors = {'Without': 'tab:blue', 'With Memory Zone': 'tab:cyan', 'With Zone + Pipes': 'tab:green'}
    vocab_colors = {'Without': 'tab:red', 'With Memory Zone': 'tab:orange', 'With Zone + Pipes': 'tab:purple'}
    
    for result in results:
        test_key = result['test_name']
        if 'Zone + Pipes' in test_key:
            test_key = 'With Zone + Pipes'
        elif 'With Memory Zone' in test_key:
            test_key = 'With Memory Zone'
        else:
            test_key = 'Without'

        ax1.plot(result['batches'], result['memory_mb'], label=f"{result['test_name']} (Memory)", color=mem_colors.get(test_key), marker='.')
        ax2.plot(result['batches'], result['vocab_size'], label=f"{result['test_name']} (Vocab)", color=vocab_colors.get(test_key), linestyle='--')
        
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='center right')

    # --- Plot 2: Final Memory Growth Comparison ---
    ax3 = axes[1]
    ax3.set_title('Final Memory Growth Comparison')
    test_names = [r['test_name'] for r in results]
    memory_growths = [r['memory_growth'] for r in results]
    colors = ['tab:red', 'tab:orange', 'tab:green'] # Added 3rd color
    
    bars = ax3.bar(test_names, memory_growths, color=colors)
    ax3.set_ylabel('Total Memory Growth (MB)')
    
    for bar in bars:
        height = bar.get_height()
        ax3.annotate(f'{height:.1f} MB',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),
                     textcoords="offset points",
                     ha='center', va='bottom')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    report_filename = os.path.join(output_path, 'memory_consumption_report.png')
    plt.savefig(report_filename, dpi=300)
    plt.close()
    print(f"Visual report saved to: {report_filename}")


def run_test(ner_model: SpacyModel, df: pd.DataFrame, batch_size: int, test_name: str, output_csv_path: str):
    """Runs a test scenario, processes data in batches, and profiles memory."""
    print(f"\n--- Running Test: {test_name} ---")
    print(f"--- Config: use_memory_zone={ner_model.use_memory_zones}, use_select_pipes={ner_model.use_select_pipes} ---")
    
    profiler = MemoryProfiler()
    initial_memory = profiler.get_memory_mb()
    
    memory_data, vocab_data, string_store_data, batch_numbers = [], [], [], []
    
    first_batch = True
    num_batches = (len(df) + batch_size - 1) // batch_size
    for i in range(0, len(df), batch_size):
        batch_df = df.iloc[i:i+batch_size]
        
        dsp = batch_df['dsp'].to_list()
        mmo = batch_df['mmo'].to_list()
        dsp_vec = np.array(dsp, dtype='|S0').reshape(len(dsp), 1)
        mmo_vec = np.array(mmo, dtype='|S0').reshape(len(mmo), 1)
        
        # Assuming you have a 'Request' class available in your scope
        requests = [Request(inputs=[
            Tensor(data=dsp_vec, name='dsp'),
            Tensor(data=mmo_vec, name='mmo')
        ])]
        
        # This calls ner.execute()
        raw_results = ner_model.execute(requests)
        
        outputs = []
        for raw_result in raw_results:
            # Assuming output_tensors() returns 3 tensors
            tensors = raw_result.output_tensors()
            if len(tensors) != 3:
                print(f"Warning: Expected 3 output tensors, got {len(tensors)}")
                continue

            labels, extracted_texts, entity_ids = tensors
            labels = labels.as_numpy().tolist()
            extracted_texts = extracted_texts.as_numpy().tolist()
            entity_ids = entity_ids.as_numpy().tolist()
            
            for label_list, extracted_text_list, entity_id_list in zip(labels, extracted_texts, entity_ids):
                these_outputs = []
                decoded_labels = [x.decode('utf-8') for x in label_list]
                decoded_entity_ids = [x.decode('utf-8') for x in entity_id_list]
                for label, extracted_text, entity_id in zip(decoded_labels, extracted_text_list, decoded_entity_ids):
                    if label:
                        these_outputs.append({
                            'entity_type': label,
                            'extracted_entity': extracted_text,
                            'standardized_entity': entity_id
                        })
                outputs.append(these_outputs)
        
        batch_df['outputs_ner'] = outputs
        
        if first_batch:
            batch_df.to_csv(output_csv_path, index=False, mode='w')
            first_batch = False
        else:
            batch_df.to_csv(output_csv_path, index=False, mode='a', header=False)
        
        batch_num = (i // batch_size) + 1
        memory_mb = profiler.get_memory_mb()
        vocab_stats = profiler.get_vocab_stats(ner_model.nlp)
        
        memory_data.append(memory_mb)
        vocab_data.append(vocab_stats['vocab_size'])
        string_store_data.append(vocab_stats['string_store_size'])
        batch_numbers.append(batch_num)
        
        print(
            f"  Batch {batch_num}/{num_batches}: "
            f"Memory = {memory_mb:.2f} MB | "
            f"Vocab Size = {vocab_stats['vocab_size']} | "
            f"String Store = {vocab_stats['string_store_size']}"
        )
        
        del batch_df, dsp, mmo, requests, raw_results, outputs
        gc.collect()
        time.sleep(0.05)
        
    final_memory = profiler.get_memory_mb()
    print(f"Test complete. Scored output saved to: {output_csv_path}")
    
    return {
        'test_name': test_name,
        'memory_mb': memory_data,
        'vocab_size': vocab_data,
        'string_store_size': string_store_data,
        'batches': batch_numbers,
        'memory_growth': final_memory - initial_memory,
    }

# --- Main Execution Block ---

def main():
    # --------------------------------------------------------------------
    # SCRIPT CONFIGURATION
    # --------------------------------------------------------------------
    # 1. Set the batch size for processing records.
    BATCH_SIZE = 500

    INPUT_CSV = "./case_examples.csv"
    
    OUTPUT_DIR = "./output"
    # --------------------------------------------------------------------
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Initialize Model
    # This will now add 'doc_cleaner' by default
    ner = SpacyModel()
    init_args = {'model_name': 'modelr'}
    ner.initialize(init_args)
    
    # Load and Prepare Data
    df = pd.read_csv(INPUT_CSV, keep_default_na=False)
    df['dsp'] = df['dsp'].apply(unidecode)
    df['mmo'] = df['mmo'].apply(unidecode)
    print(f"Loaded {len(df)} records to be processed in batches of {BATCH_SIZE}.")
    
    # Run Investigation
    all_results = []
    
    # TEST 1: Without Memory Zone (Baseline)
    ner.use_memory_zones = False
    ner.use_select_pipes = False
    results_without_zone = run_test(
        ner, df, BATCH_SIZE, "Without Memory Zone", 
        os.path.join(OUTPUT_DIR, "output_without_zone.csv")
    )
    all_results.append(results_without_zone)
    
    time.sleep(2)
    gc.collect()
    
    # TEST 2: With Memory Zone Only
    ner.use_memory_zones = True
    ner.use_select_pipes = False
    results_with_zone = run_test(
        ner, df, BATCH_SIZE, "With Memory Zone",
        os.path.join(OUTPUT_DIR, "output_with_zone.csv")
    )
    all_results.append(results_with_zone)
    
    time.sleep(2)
    gc.collect()

    # === NEW: TEST 3: With Memory Zone + Select Pipes (Full Optimization) ===
    ner.use_memory_zones = True
    ner.use_select_pipes = True # <-- This is the new flag
    results_with_zone_and_pipes = run_test(
        ner, df, BATCH_SIZE, "With Zone + Pipes",
        os.path.join(OUTPUT_DIR, "output_with_zone_and_pipes.csv")
    )
    all_results.append(results_with_zone_and_pipes)
    # === End of new test ===
    
    # Generate Final Report
    generate_report(all_results, OUTPUT_DIR)
    
    print("\nInvestigation Finished.")

if __name__ == "__main__":
    main()
