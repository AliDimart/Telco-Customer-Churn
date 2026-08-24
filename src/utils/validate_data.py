import great_expectations as gx
from typing import Tuple, List


def validate_telco_data(df) -> Tuple[bool, List[str]]:
    """
    Comprehensive data validation for Telco Customer Churn dataset using Great Expectations.
    
    This function implements critical data quality checks that must pass before model training.
    It validates data integrity, business logic constraints, and statistical properties
    that the ML model expects.

    Returns
    -------
    success : bool
        True if all validation checks passed.
    failed_expectations : list[str]
        Names of failed checks.
    
    """
    print("🔍 Starting data validation with Great Expectations...")

    failed_expectations: List[str] = []
    
    required_columns = [ 
        "customerID", "gender", "Partner", "Dependents", "PhoneService", "InternetService", "Contract", "tenure", "MonthlyCharges", "TotalCharges",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "PaperlessBilling" 
    ]

    missing_columns = [column for column in required_columns if column not in df.columns]

    if missing_columns:
        failed_expectations.append(f"missing_columns: {missing_columns}")

        return False, failed_expectations

    # CREATE GREAT EXPECTATIONS BATCH
    context = gx.get_context(mode="ephemeral")

    data_source = context.data_sources.add_pandas(name="telco_data_source")

    data_asset = data_source.add_dataframe_asset(name="telco_dataframe")

    batch_definition = data_asset.add_batch_definition_whole_dataframe("telco_batch")

    batch = batch_definition.get_batch(batch_parameters={"dataframe": df})

    # DEFINE EXPECTATIONS

    expectations = []

    expectations.append(gx.expectations.ExpectColumnValuesToBeInSet(column="gender", value_set=["Male", "Female"]))

    yes_no_columns = [
        "Partner", "Dependents", "PhoneService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "PaperlessBilling"
    ]

    for column in yes_no_columns:
        expectations.append(gx.expectations.ExpectColumnValuesToBeInSet(column=column, value_set=["Yes", "No"]))

    if "Churn" in df.columns:
        expectations.append(gx.expectations.ExpectColumnValuesToNotBeNull(column='Churn'))
        expectations.append(gx.expectations.ExpectColumnValuesToBeInSet(column="Churn", value_set=["Yes", "No"]))
        
    expectations.append(gx.expectations.ExpectColumnValuesToBeInSet(column='Contract', value_set=['Month-to-month', 'One year', 'Two year']))
    expectations.append(gx.expectations.ExpectColumnValuesToBeInSet(column='InternetService', value_set=['DSL', 'Fiber optic', 'No']))
    
    expectations.append(gx.expectations.ExpectColumnValuesToBeBetween(column='tenure', min_value=0))
    expectations.append(gx.expectations.ExpectColumnValuesToBeBetween(column='MonthlyCharges', min_value=0))
    expectations.append(gx.expectations.ExpectColumnValuesToBeBetween(column='TotalCharges', min_value=0))

    expectations.append(gx.expectations.ExpectColumnValuesToNotBeNull(column='tenure'))
    expectations.append(gx.expectations.ExpectColumnValuesToNotBeNull(column='MonthlyCharges'))

    expectations.append(gx.expectations.ExpectColumnValuesToNotBeNull(column='customerID'))
    expectations.append(gx.expectations.ExpectColumnValuesToBeUnique(column="customerID"))

    # RUN EXPECTATIONS
    
    print("   ⚙️ Running validation suite...")

    for expectation in expectations:

        result = batch.validate(expectation)

        if not result.success:
        
            expectation_type = result.expectation_config.type

            failed_expectations.append(expectation_type)

    # FINAL RESULT

    success = len(failed_expectations) == 0

    if success:
        print("✅ Data validation PASSED")
    else:
        print("❌ Data validation FAILED")
        print("   Failed checks:")

        for failure in failed_expectations:
            print(f"      - {failure}")

    return success, failed_expectations