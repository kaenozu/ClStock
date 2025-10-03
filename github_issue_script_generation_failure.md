# GitHub Issue: Script Generation Failure - NameError for 'symbol' Variable

## Issue Title
Fix NameError in data retrieval script generation - 'symbol' variable not defined

## Description
A bug has been identified in the data retrieval script generation mechanism that causes a script generation failure. The error occurs when trying to create a Google Colab helper script for symbols that failed to download.

## Error Details
- **Error Type**: `NameError: name 'symbol' is not defined`
- **File**: `data_retrieval_script_generator.py`
- **Line**: 43
- **Method**: `_generate_data_retrieval_script` (called from `FullAutoInvestmentSystem._generate_data_retrieval_script`)

## Specific Failure Case
- **Failed Symbol**: 6502
- The variable name doesn't match between the definition and usage in the script generation function

## Root Cause
The bug occurs in the script generation code where there's a mismatch between variable naming in the generator function. The template string for the data retrieval script references the variable `symbol` which is not properly defined in the generated script's scope.

## Expected Solution
Need to fix the variable reference in the generator function in `data_retrieval_script_generator.py` to ensure proper variable scoping in the generated script.

## Affected Files
1. `data_retrieval_script_generator.py` - Contains the script generation logic
2. `full_auto_system.py` - Contains the `_generate_data_retrieval_script` method that calls the generator

## Impact
This bug prevents the automatic generation of Google Colab helper scripts for manual data retrieval when symbol data acquisition fails, hindering the system's ability to recover from data download failures.