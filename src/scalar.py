import pandas as pd
import os

def run_normalization(input_file="engineered_features.csv", output_file="normalized_data.csv"):
    print(f"\n=== Scalar Normalization (Module: scalar.py) ===")
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return
        
    df = pd.read_csv(input_file)
    
    # Using 'temp' from engineered_data instead of 'Average_Lifetime_Temperature'
    if "temp" in df.columns:
        temp_col = "temp"
    else:
        # Fallback if names differ
        temp_cols = df.filter(like='temp').columns
        if len(temp_cols) > 0:
            temp_col = temp_cols[0]
        else:
            print("Error: Temperature column not found.")
            return

    print(f"  Calculating Temp_Score based on column: {temp_col}")
    
    t_min = df[temp_col].min()
    t_max = df[temp_col].max()
    df["Temp_Score"] = 1 - (df[temp_col] - t_min) / (t_max - t_min + 1)

    # Temperature-only scoring (as per your modular code)
    weight_temp = 1.0

    df["Final_Score"] = weight_temp * df["Temp_Score"]
    df["Efficiency_Score"] = df["Final_Score"] * 100

    ranked_df = df.sort_values(by="Final_Score", ascending=False)
    
    # Save the normalized data
    ranked_df.to_csv(output_file, index=False)
    print(f"  Normalization complete. Saved to {output_file}")
    
    # Print sample
    print("\n  Sample Scores (Top 5):")
    cols_to_show = ["rectifier_id", "Final_Score", "Efficiency_Score"]
    existing_cols = [c for c in cols_to_show if c in ranked_df.columns]
    print(ranked_df[existing_cols].head())
    
    return output_file

if __name__ == "__main__":
    run_normalization()