## 1. Initialise Venv(if not done)
python -m venv .venv

## 2. Activate Venv
.\.venv\Scripts\Activate.ps1

or

.venv\Scripts\activate.bat

## 3. Install dependencies (if not done)
python -m pip install streamlit pandas numpy plotly openpyxl

## 4. Run the application
streamlit run cycling_workout_analyzer.py

## Working with blocks
- Upload a ride CSV and confirm the suggested column mappings.
- Select a region on the timeline to create a block, or add rows in the block table.
- Add or remove rows in the block table, edit their name, boundaries, type, and notes, then click **Save block edits**. Save before using other controls so all cell changes are committed. To adjust a block visually, choose it in **Adjust a block by dragging its time range**, drag the two handles, then apply the range.
- Set a ride name for the CSV and Excel export filenames, then choose metrics and analyze.
- Save a named analysis setup under **Save current setup**. Setups are saved locally in .cycling_analysis_setups.json beside the app and reload after restart. Use **Export setup library** and **Import setup library (JSON)** to move them to another computer. Setups contain configuration and block templates, never ride data.
- Save/load independent block templates under **Reusable block templates**. They persist locally in .cycling_block_templates.json and can also be imported/exported as JSON.
- Use **Detect intervals from power** to propose sustained above-threshold power ranges. Tune smoothing, minimum duration, and short-gap merging; choose whether to replace or append blocks, then review the candidates in the block editor.
- In block analysis, choose Bar, Line, Area, or Points for the selected metric. Use **Block time-series overlays** to compare one stream across selected blocks, aligned either to ride time or each block's start. Choose **All blocks** in the zone scope for a stacked block-by-zone comparison.
- Set the ride date before analyzing. Each analysis updates the local ride history with ride-level and block-level metrics; view ride, weekly-average, or monthly-average progression in **Ride history and calendar progression**. Export/import history JSON to back it up or move it to another computer.

## Next iterations
1. ✅ Save and reload named analysis setups (column mappings, FTP, metrics, and block templates).
2. ✅ Add reusable block templates and optional automatic interval detection.
3. ✅ Add chart choices per metric, including time-series overlays and block-by-block zone distributions.
4. ✅ Store ride history and show calendar-based ride, weekly, and monthly progression graphs.
5. Add interval quality checks for gaps, duplicate timestamps, sensor dropouts, and noisy elevation.
6. Compare rides with deeper cohort and training summaries.
