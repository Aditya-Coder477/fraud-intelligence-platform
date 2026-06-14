import traceback
import sys

def main():
    log_file = open("execution_log.txt", "w")
    def log(msg):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    log("=== STARTING COMPLETE MULE ACCOUNT ANALYSIS AND NOTEBOOK GENERATION ===")
    
    # Run detailed analysis
    log("\n[Step 1] Running detailed data analysis...")
    try:
        import detailed_analysis
        detailed_analysis.run_detailed_analysis("DataSet.csv")
        log("Detailed analysis completed successfully!")
    except Exception as e:
        log(f"Error during detailed analysis: {e}")
        log(traceback.format_exc())
        
    # Generate EDA notebook
    log("\n[Step 2] Generating Jupyter Notebook...")
    try:
        import create_notebook
        create_notebook.generate_notebook()
        log("Jupyter Notebook generation completed successfully!")
    except Exception as e:
        log(f"Error during notebook generation: {e}")
        log(traceback.format_exc())
        
    log("\n=== ALL STEPS COMPLETED ===")
    log_file.close()

if __name__ == "__main__":
    main()
