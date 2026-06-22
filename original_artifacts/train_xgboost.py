import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
from cpmpy.transformations.get_variables import get_variables
from sklearn.ensemble import RandomForestClassifier
from feature_extraction import extract_constraint_features

# Use the new cleaned dataset with only the original features
csv_file = "synthetic_constraints_with_original_features.csv"
df = pd.read_csv(csv_file)

def get_feature_columns(df):
    """Get all columns except model and label which are used as features for training"""
    return [col for col in df.columns if col not in ["model", "label"]]

X = df[get_feature_columns(df)]
y = df["label"].astype(int)

# Print information about the features
print(f"Total features: {len(X.columns)}")
print("Feature categories:")
print(f"- Using ONLY the 14 ORIGINAL FEATURES from the paper")
print(f"- 1. Relation (String): Constraint relation type")
print(f"- 2. Arity (Int): Number of variables in the constraint")
print(f"- 3. Has constant (Bool): Whether a constant value appears")
print(f"- 4. Constant (Int): The constant value if present")
print(f"- 5. Var name same (Bool): If all variables share same name")
print(f"- 6. Var Ndims same (Bool): If all variables have same number of dimensions")
print(f"- 7. Var Ndims max (Int): Maximum dimension count among variables")
print(f"- 8. Var Ndims min (Int): Minimum dimension count among variables")
print(f"- 9. Var dimi has (Bool): If dimension i is present for all variables")
print(f"- 10. Var dimi same (Bool): If dimension i is same for all variables")
print(f"- 11. Var dimi max (Int): Maximum value for dimension i")
print(f"- 12. Var dimi min (Int): Minimum value for dimension i")
print(f"- 13. Var dimi avg (Float): Average value for dimension i")
print(f"- 14. Var dimi spread (Float): Spread of dimension i values")

categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
if categorical_cols:
    print(f"Encoding categorical columns: {categorical_cols}")
    X = pd.get_dummies(X, columns=categorical_cols)

# Save the feature columns for later use in prediction
with open("xgb_feature_columns.txt", "w") as f:
    for col in X.columns:
        f.write(col + "\n")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost classifier
print("\nTraining XGBoost model...")
clf = xgb.XGBClassifier(eval_metric='logloss', use_label_encoder=False)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
joblib.dump(clf, "constraint_classifier_xgb.joblib")

# Get feature importance
feature_importance = clf.feature_importances_
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importance
})
importance_df = importance_df.sort_values(by='Importance', ascending=False)
print("\nTop 20 most important features:")
print(importance_df.head(20))

# Train Random Forest classifier for comparison
print("\nTraining Random Forest model...")
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)
y_pred_rf = rf_clf.predict(X_test)
print("[RandomForest] Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))
joblib.dump(rf_clf, "constraint_classifier_rf.joblib")

# Function to evaluate constraints using the trained models
def evaluate_constraints(model, X_columns, constraints, variables, model_name):
    print(f"\n--- Constraint Validity Probabilities [{model_name}] ---")
    for c in constraints:
        if "sum" in str(c).lower() or "alldifferent" in str(c).lower() or "count" in str(c).lower():
            feats = extract_constraint_features(c, variables, all_constraints=constraints)
            feat_df = pd.DataFrame([feats])
            feat_df = feat_df.reindex(columns=X_columns, fill_value=0)
            prob = model.predict_proba(feat_df)[0][1]
            print(f"[{model_name}] Constraint {c}: Probability of being valid = {prob:.4f}")
            
            # Print the top 5 features that contribute to this prediction
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                feature_imp = list(zip(X_columns, importances))
                feature_imp.sort(key=lambda x: x[1], reverse=True)
                print(f"  Top features for this constraint:")
                for feat, imp in feature_imp[:5]:
                    feat_value = feat_df[feat].values[0]
                    print(f"  - {feat}={feat_value:.4f} (importance={imp:.4f})")

# Test on benchmark problems
import sys
sys.path.append("benchmarks_global")
from sudoku import construct_sudoku

# Evaluate constraints on Sudoku
instance, oracle = construct_sudoku(3, 3, 9)
variables = instance.variables
model_constraints = oracle.constraints
print("\n--- Sudoku Constraints ---")
evaluate_constraints(clf, X.columns, model_constraints, variables, "XGBoost")
evaluate_constraints(rf_clf, X.columns, model_constraints, variables, "RandomForest")
input("Press Enter to continue...")

# Try to evaluate on other benchmarks
try:
    from benchmarks_global.uefa import construct_uefa as construct_uefa_global
    teams_data = {
        "Real Madrid": {"country": "ESP", "coefficient": 134000},
        "Bayern Munich": {"country": "GER", "coefficient": 129000},
        "Manchester City": {"country": "ENG", "coefficient": 128000},
        "PSG": {"country": "FRA", "coefficient": 112000},
        "Liverpool": {"country": "ENG", "coefficient": 109000},
        "Barcelona": {"country": "ESP", "coefficient": 98000},
        "Juventus": {"country": "ITA", "coefficient": 95000},
        "Atletico Madrid": {"country": "ESP", "coefficient": 94000},
        "Manchester United": {"country": "ENG", "coefficient": 92000},
        "Chelsea": {"country": "ENG", "coefficient": 91000},
        "Borussia Dortmund": {"country": "GER", "coefficient": 88000},
        "Ajax": {"country": "NED", "coefficient": 82000},
        "RB Leipzig": {"country": "GER", "coefficient": 79000},
        "Inter Milan": {"country": "ITA", "coefficient": 76000},
        "Sevilla": {"country": "ESP", "coefficient": 75000},
        "Napoli": {"country": "ITA", "coefficient": 74000},
        "Benfica": {"country": "POR", "coefficient": 73000},
        "Porto": {"country": "POR", "coefficient": 72000},
        "Olympiacos": {"country": "GRE", "coefficient": 66000},
        "Celtic": {"country": "SCO", "coefficient": 65000},
        "Rangers": {"country": "SCO", "coefficient": 64000},
        "PSV Eindhoven": {"country": "NED", "coefficient": 63000},
        "Sporting CP": {"country": "POR", "coefficient": 62000},
        "Marseille": {"country": "FRA", "coefficient": 61000},
        "Club Brugge": {"country": "BEL", "coefficient": 60000},
        "Galatasaray": {"country": "TUR", "coefficient": 59000},
        "Feyenoord": {"country": "NED", "coefficient": 58000}
    }
    instance, oracle = construct_uefa_global(teams_data)
    variables = instance.variables
    model_constraints = oracle.constraints
    print("\n--- UEFA Constraints ---")
    evaluate_constraints(clf, X.columns, model_constraints, variables, "XGBoost")
    evaluate_constraints(rf_clf, X.columns, model_constraints, variables, "RandomForest")
except Exception as e:
    print(f"Error evaluating UEFA: {e}")

try:
    from benchmarks_global.examtt import ces_global
    instance, oracle = ces_global(nsemesters=9, courses_per_semester=6, slots_per_day=9, days_for_exams=14)
    variables = instance.variables
    model_constraints = oracle.constraints
    print("\n--- Exam Timetabling Constraints ---")
    evaluate_constraints(clf, X.columns, model_constraints, variables, "XGBoost")
    evaluate_constraints(rf_clf, X.columns, model_constraints, variables, "RandomForest")
except Exception as e:
    print(f"Error evaluating Exam Timetabling: {e}")

try:
    from benchmarks_global.vm_allocation import construct_vm_allocation
    from vm_allocation_model import PM_DATA, VM_DATA
    instance, oracle = construct_vm_allocation(PM_DATA, VM_DATA)
    variables = instance.variables
    model_constraints = oracle.constraints
    print("\n--- VM Allocation Constraints ---")
    evaluate_constraints(clf, X.columns, model_constraints, variables, "XGBoost")
    evaluate_constraints(rf_clf, X.columns, model_constraints, variables, "RandomForest")
except Exception as e:
    print(f"Error evaluating VM Allocation: {e}")
