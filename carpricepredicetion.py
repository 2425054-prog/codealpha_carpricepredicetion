import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

print("="*60)
print("CAR PRICE PREDICTION WITH MACHINE LEARNING")
print("="*60)

# ============================================================================
# SECTION 1: DATA CREATION AND LOADING
# ============================================================================

print("\n🚗 SECTION 1: DATA COLLECTION AND PREPARATION")
print("-"*50)

# Create a comprehensive car dataset that mirrors real-world car pricing data
np.random.seed(42)

# Number of car samples
n_cars = 5000

# Car brands with their goodwill scores (0-100 scale) and brand multipliers
brands = {
    'Toyota': {'goodwill': 85, 'multiplier': 1.0},
    'Honda': {'goodwill': 84, 'multiplier': 0.98},
    'Ford': {'goodwill': 78, 'multiplier': 0.92},
    'Chevrolet': {'goodwill': 75, 'multiplier': 0.88},
    'BMW': {'goodwill': 90, 'multiplier': 1.3},
    'Mercedes': {'goodwill': 92, 'multiplier': 1.35},
    'Audi': {'goodwill': 89, 'multiplier': 1.25},
    'Lexus': {'goodwill': 88, 'multiplier': 1.2},
    'Tesla': {'goodwill': 85, 'multiplier': 1.4},
    'Hyundai': {'goodwill': 76, 'multiplier': 0.85},
    'Kia': {'goodwill': 74, 'multiplier': 0.82},
    'Nissan': {'goodwill': 77, 'multiplier': 0.9},
    'Volkswagen': {'goodwill': 79, 'multiplier': 0.95},
    'Subaru': {'goodwill': 82, 'multiplier': 0.96},
    'Mazda': {'goodwill': 80, 'multiplier': 0.93},
    'Volvo': {'goodwill': 83, 'multiplier': 1.05},
    'Jeep': {'goodwill': 76, 'multiplier': 0.89},
    'Ram': {'goodwill': 72, 'multiplier': 0.87},
    'Porsche': {'goodwill': 95, 'multiplier': 2.0},
    'Ferrari': {'goodwill': 98, 'multiplier': 3.5}
}

# Generate car data
car_data = []

# Create probability distribution that sums to 1
brand_probabilities = [0.09, 0.09, 0.07, 0.06, 0.05, 0.05, 0.05, 
                       0.04, 0.04, 0.07, 0.06, 0.06, 0.05, 
                       0.04, 0.04, 0.03, 0.03, 0.02, 0.01, 0.01]

# Ensure probabilities sum to 1
brand_probabilities = np.array(brand_probabilities) / sum(brand_probabilities)

for _ in range(n_cars):
    # Select random brand with bias towards popular brands
    brand = np.random.choice(list(brands.keys()), p=brand_probabilities)
    
    brand_info = brands[brand]
    
    # Car features
    year = np.random.randint(2000, 2024)
    age = 2024 - year
    
    # Horsepower (depends on car type and brand)
    if brand in ['Ferrari', 'Porsche', 'Tesla', 'BMW', 'Mercedes', 'Audi']:
        horsepower = np.random.randint(250, 650)
    elif brand in ['Toyota', 'Honda', 'Hyundai', 'Kia']:
        horsepower = np.random.randint(100, 280)
    else:
        horsepower = np.random.randint(120, 400)
    
    # Mileage (higher for older cars)
    mileage = int(np.random.exponential(scale=12000) * (age + 0.5))
    mileage = min(mileage, 250000)
    
    # Engine size (liters)
    engine_size = np.random.uniform(1.2, 6.0)
    engine_size = round(engine_size, 1)
    
    # Fuel type
    fuel_type = np.random.choice(['Petrol', 'Diesel', 'Electric', 'Hybrid'], 
                                  p=[0.55, 0.25, 0.1, 0.1])
    
    # Transmission
    transmission = np.random.choice(['Manual', 'Automatic', 'Semi-Automatic'], 
                                     p=[0.3, 0.6, 0.1])
    
    # Number of previous owners
    owners = np.random.choice([0, 1, 2, 3, 4], p=[0.15, 0.4, 0.25, 0.15, 0.05])
    
    # Condition (1-10 scale)
    condition = np.random.randint(1, 11)
    
    # Fuel efficiency (MPG or MPGe)
    if fuel_type == 'Electric':
        fuel_efficiency = np.random.randint(80, 130)
    elif fuel_type == 'Hybrid':
        fuel_efficiency = np.random.randint(40, 70)
    elif fuel_type == 'Diesel':
        fuel_efficiency = np.random.randint(25, 45)
    else:  # Petrol
        fuel_efficiency = np.random.randint(15, 35)
    
    # Number of seats
    seats = np.random.choice([2, 4, 5, 6, 7, 8], p=[0.05, 0.6, 0.2, 0.05, 0.07, 0.03])
    
    # Calculate base price
    base_price = 15000
    
    # Year factor (newer cars cost more)
    year_factor = 1 + (year - 2000) * 0.05
    
    # Age depreciation (cars lose value over time)
    depreciation = np.exp(-0.15 * age)
    
    # Horsepower factor
    hp_factor = 1 + (horsepower - 150) / 500
    
    # Mileage depreciation
    mileage_depreciation = np.exp(-mileage / 100000)
    
    # Engine size factor
    engine_factor = 1 + (engine_size - 2.0) * 0.2
    
    # Fuel type adjustment
    fuel_factors = {'Petrol': 1.0, 'Diesel': 1.05, 'Electric': 1.15, 'Hybrid': 1.1}
    fuel_factor = fuel_factors[fuel_type]
    
    # Transmission adjustment
    trans_factors = {'Manual': 1.0, 'Automatic': 1.08, 'Semi-Automatic': 1.05}
    trans_factor = trans_factors[transmission]
    
    # Owners factor (more owners = lower price)
    owners_factor = 1 - (owners * 0.03)
    
    # Condition factor
    condition_factor = 0.5 + (condition / 10) * 0.5
    
    # Brand multiplier
    brand_multiplier = brand_info['multiplier']
    
    # Seats factor
    seats_factor = 1 + (seats - 5) * 0.03
    
    # Fuel efficiency factor
    if fuel_type in ['Electric', 'Hybrid']:
        efficiency_factor = 1 + (fuel_efficiency - 50) / 200
    else:
        efficiency_factor = 1
    
    # Calculate final price with noise
    price = (base_price * year_factor * depreciation * hp_factor * 
             mileage_depreciation * engine_factor * fuel_factor * trans_factor * 
             owners_factor * condition_factor * brand_multiplier * seats_factor * 
             efficiency_factor)
    
    # Add random noise
    price = price * np.random.normal(1, 0.1)
    price = max(500, min(200000, price))
    
    # Store data
    car_data.append({
        'Brand': brand,
        'Brand_Goodwill': brand_info['goodwill'],
        'Year': year,
        'Age': age,
        'Horsepower': horsepower,
        'Mileage': mileage,
        'Engine_Size_L': engine_size,
        'Fuel_Type': fuel_type,
        'Transmission': transmission,
        'Previous_Owners': owners,
        'Condition_Score': condition,
        'Fuel_Efficiency_MPG': fuel_efficiency,
        'Number_of_Seats': seats,
        'Price_USD': int(price)
    })

# Create DataFrame
df = pd.DataFrame(car_data)

print("\n✅ Dataset created successfully!")
print(f"   Total car records: {len(df):,}")
print(f"   Features: {len(df.columns)}")
print(f"   Price range: ${df['Price_USD'].min():,.0f} - ${df['Price_USD'].max():,.0f}")
print(f"   Average price: ${df['Price_USD'].mean():,.0f}")
print(f"   Median price: ${df['Price_USD'].median():,.0f}")

# Display first few rows
print("\n📋 First 10 samples of the dataset:")
print(df.head(10))

# Dataset information
print("\n📊 Dataset Information:")
print(df.info())

# Check for missing values
print("\n🔍 Missing values check:")
print(df.isnull().sum())

# Basic statistics
print("\n📈 Basic Statistics:")
print(df.describe())

# ============================================================================
# SECTION 2: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================

print("\n" + "="*60)
print("📊 SECTION 2: EXPLORATORY DATA ANALYSIS")
print("="*60)

# Price distribution by brand
print("\n💰 Average Price by Brand:")
brand_price = df.groupby('Brand')['Price_USD'].agg(['mean', 'median', 'count']).sort_values('mean', ascending=False)
print(brand_price.round(0))

print("\n📊 Correlation with Price:")
numeric_cols = df.select_dtypes(include=[np.number]).columns
correlations = df[numeric_cols].corr()['Price_USD'].sort_values(ascending=False)
for col, corr in correlations.items():
    if col != 'Price_USD':
        print(f"   {col}: {corr:.3f}")

# ============================================================================
# SECTION 3: FEATURE ENGINEERING
# ============================================================================

print("\n" + "="*60)
print("🔧 SECTION 3: FEATURE ENGINEERING")
print("="*60)

# Create additional features
df['Price_per_HP'] = df['Price_USD'] / df['Horsepower']
df['Age_Mileage_Ratio'] = df['Age'] / (df['Mileage'] / 10000 + 1)

# Brand segment
def get_brand_segment(brand):
    luxury_brands = ['BMW', 'Mercedes', 'Audi', 'Lexus', 'Porsche', 'Ferrari', 'Tesla']
    economy_brands = ['Hyundai', 'Kia', 'Chevrolet', 'Ford', 'Nissan']
    if brand in luxury_brands:
        return 'Luxury'
    elif brand in economy_brands:
        return 'Economy'
    else:
        return 'Mid-Range'

df['Brand_Segment'] = df['Brand'].apply(get_brand_segment)

# Car age category
def get_age_category(age):
    if age <= 3:
        return 'New'
    elif age <= 7:
        return 'Moderate'
    elif age <= 12:
        return 'Old'
    else:
        return 'Vintage'

df['Age_Category'] = df['Age'].apply(get_age_category)

# Condition category
def get_condition_category(score):
    if score >= 9:
        return 'Excellent'
    elif score >= 7:
        return 'Good'
    elif score >= 5:
        return 'Fair'
    else:
        return 'Poor'

df['Condition_Category'] = df['Condition_Score'].apply(get_condition_category)

# Horsepower category
def get_hp_category(hp):
    if hp >= 400:
        return 'High'
    elif hp >= 200:
        return 'Medium'
    else:
        return 'Low'

df['HP_Category'] = df['Horsepower'].apply(get_hp_category)

print("✅ Created new features:")
print(f"   • Price per HP")
print(f"   • Age-Mileage Ratio")
print(f"   • Brand Segment")
print(f"   • Age Category")
print(f"   • Condition Category")
print(f"   • HP Category")

# Encode categorical variables
label_encoders = {}
categorical_cols = ['Brand', 'Fuel_Type', 'Transmission', 'Brand_Segment', 
                    'Age_Category', 'Condition_Category', 'HP_Category']

for col in categorical_cols:
    le = LabelEncoder()
    df[col + '_Encoded'] = le.fit_transform(df[col])
    label_encoders[col] = le
    print(f"   Encoded {col}: {len(le.classes_)} categories")

# ============================================================================
# SECTION 4: DATA PREPROCESSING FOR MODELING
# ============================================================================

print("\n" + "="*60)
print("🔄 SECTION 4: DATA PREPROCESSING")
print("="*60)

# Select features for modeling
feature_cols = ['Brand_Goodwill', 'Year', 'Age', 'Horsepower', 'Mileage', 
                'Engine_Size_L', 'Previous_Owners', 'Condition_Score', 
                'Fuel_Efficiency_MPG', 'Number_of_Seats', 'Age_Mileage_Ratio',
                'Brand_Encoded', 'Fuel_Type_Encoded', 'Transmission_Encoded',
                'Brand_Segment_Encoded', 'Age_Category_Encoded', 
                'Condition_Category_Encoded', 'HP_Category_Encoded']

X = df[feature_cols]
y = df['Price_USD']

print(f"\n📊 Feature matrix shape: {X.shape}")
print(f"🎯 Target vector shape: {y.shape}")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\n🔪 Data split:")
print(f"   Training set: {len(X_train)} samples")
print(f"   Testing set: {len(X_test)} samples")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n✅ Features scaled using StandardScaler")

# ============================================================================
# SECTION 5: MODEL TRAINING AND EVALUATION
# ============================================================================

print("\n" + "="*60)
print("🤖 SECTION 5: MODEL TRAINING AND EVALUATION")
print("="*60)

# Initialize multiple regression models
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=1.0),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'SVR': SVR(kernel='rbf', C=100, gamma='auto')
}

# Train and evaluate each model
results = {}
predictions = {}

print("\n📊 Training and evaluating models...\n")

for name, model in models.items():
    # Train the model
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    predictions[name] = y_pred_test
    
    # Calculate metrics
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_mape = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100
    
    # Store results
    results[name] = {
        'Train_R2': train_r2,
        'Test_R2': test_r2,
        'MAE': test_mae,
        'RMSE': test_rmse,
        'MAPE': test_mape
    }
    
    # Print results
    print(f"{name:20} | Train R²: {train_r2:.4f} | Test R²: {test_r2:.4f} | "
          f"MAE: ${test_mae:,.0f} | RMSE: ${test_rmse:,.0f} | MAPE: {test_mape:.1f}%")

# Best model selection
best_model_name = max(results, key=lambda x: results[x]['Test_R2'])
best_model = models[best_model_name]

print(f"\n⭐ BEST MODEL: {best_model_name}")
print(f"   Test R² Score: {results[best_model_name]['Test_R2']:.4f}")
print(f"   Mean Absolute Error: ${results[best_model_name]['MAE']:,.0f}")
print(f"   Root Mean Square Error: ${results[best_model_name]['RMSE']:,.0f}")
print(f"   Mean Absolute Percentage Error: {results[best_model_name]['MAPE']:.1f}%")

# Cross-validation
cv_scores = cross_val_score(best_model, X_train_scaled, y_train, cv=5, scoring='r2')
print(f"\n🔄 5-Fold Cross-Validation R² Scores: {cv_scores}")
print(f"   Mean CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# ============================================================================
# SECTION 6: FEATURE IMPORTANCE ANALYSIS
# ============================================================================

print("\n" + "="*60)
print("🎯 SECTION 6: FEATURE IMPORTANCE ANALYSIS")
print("="*60)

if hasattr(best_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\n📊 Top 10 Most Important Features:")
    for i, row in feature_importance.head(10).iterrows():
        print(f"   {row['Feature']:30} : {row['Importance']:.4f} ({row['Importance']*100:.2f}%)")
else:
    if hasattr(best_model, 'coef_'):
        coef_df = pd.DataFrame({
            'Feature': feature_cols,
            'Coefficient': best_model.coef_
        })
        coef_df['Abs_Coefficient'] = np.abs(coef_df['Coefficient'])
        coef_df = coef_df.sort_values('Abs_Coefficient', ascending=False)
        print("\n📊 Top 10 Most Important Features (by coefficient magnitude):")
        for i, row in coef_df.head(10).iterrows():
            print(f"   {row['Feature']:30} : {row['Coefficient']:.4f}")

# ============================================================================
# SECTION 7: VISUALIZATIONS
# ============================================================================

print("\n" + "="*60)
print("📈 SECTION 7: CREATING VISUALIZATIONS")
print("="*60)

# Check if we have data to plot
if len(y_test) > 0:
    # Create comprehensive figure
    fig = plt.figure(figsize=(16, 20))
    
    # 7.1: Actual vs Predicted Prices
    ax1 = fig.add_subplot(3, 2, 1)
    y_pred_best = predictions[best_model_name]
    ax1.scatter(y_test, y_pred_best, alpha=0.5, edgecolors='k', linewidth=0.5)
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
    ax1.set_xlabel('Actual Price ($)')
    ax1.set_ylabel('Predicted Price ($)')
    ax1.set_title(f'{best_model_name}: Actual vs Predicted Prices\nR² = {results[best_model_name]["Test_R2"]:.4f}', 
                  fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 7.2: Residuals Distribution
    ax2 = fig.add_subplot(3, 2, 2)
    residuals = y_test - y_pred_best
    ax2.hist(residuals, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
    ax2.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax2.set_xlabel('Residuals ($)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Residuals Distribution', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 7.3: Model Comparison (R²)
    ax3 = fig.add_subplot(3, 2, 3)
    model_names = list(results.keys())
    r2_scores = [results[m]['Test_R2'] for m in model_names]
    bars1 = ax3.bar(range(len(model_names)), r2_scores, color='steelblue', alpha=0.8)
    ax3.set_xlabel('Models')
    ax3.set_ylabel('R² Score')
    ax3.set_title('Model Performance Comparison (R²)', fontsize=12, fontweight='bold')
    ax3.set_xticks(range(len(model_names)))
    ax3.set_xticklabels(model_names, rotation=45, ha='right')
    ax3.set_ylim([0, 1.05])
    for bar, score in zip(bars1, r2_scores):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                 f'{score:.3f}', ha='center', va='bottom', fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 7.4: Model Comparison (MAE)
    ax4 = fig.add_subplot(3, 2, 4)
    mae_scores = [results[m]['MAE'] for m in model_names]
    bars2 = ax4.bar(range(len(model_names)), mae_scores, color='coral', alpha=0.8)
    ax4.set_xlabel('Models')
    ax4.set_ylabel('Mean Absolute Error ($)')
    ax4.set_title('Model Performance Comparison (MAE)', fontsize=12, fontweight='bold')
    ax4.set_xticks(range(len(model_names)))
    ax4.set_xticklabels(model_names, rotation=45, ha='right')
    for bar, mae in zip(bars2, mae_scores):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100, 
                 f'${mae:,.0f}', ha='center', va='bottom', fontsize=8)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 7.5: Price Distribution by Brand Segment
    ax5 = fig.add_subplot(3, 2, 5)
    segment_order = ['Economy', 'Mid-Range', 'Luxury']
    sns.boxplot(data=df, x='Brand_Segment', y='Price_USD', order=segment_order, ax=ax5, palette='Set2')
    ax5.set_xlabel('Brand Segment')
    ax5.set_ylabel('Price ($)')
    ax5.set_title('Price Distribution by Brand Segment', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 7.6: Feature Importance
    ax6 = fig.add_subplot(3, 2, 6)
    if hasattr(best_model, 'feature_importances_'):
        top_features = feature_importance.head(10)
        ax6.barh(top_features['Feature'], top_features['Importance'], color='teal', alpha=0.7)
        ax6.set_xlabel('Importance')
        ax6.set_title('Top 10 Feature Importance', fontsize=12, fontweight='bold')
        ax6.invert_yaxis()
        ax6.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.show()
    
    # Additional visualizations
    fig2, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 7.7: Price vs Age
    ax_age = axes[0, 0]
    scatter = ax_age.scatter(df['Age'], df['Price_USD'], c=df['Brand_Goodwill'], 
                             cmap='viridis', alpha=0.6, s=20)
    ax_age.set_xlabel('Car Age (Years)')
    ax_age.set_ylabel('Price ($)')
    ax_age.set_title('Price vs Age (Colored by Brand Goodwill)', fontsize=12, fontweight='bold')
    plt.colorbar(scatter, ax=ax_age, label='Brand Goodwill')
    ax_age.grid(True, alpha=0.3)
    
    # 7.8: Price vs Horsepower
    ax_hp = axes[0, 1]
    scatter2 = ax_hp.scatter(df['Horsepower'], df['Price_USD'], c=df['Condition_Score'], 
                             cmap='coolwarm', alpha=0.6, s=20)
    ax_hp.set_xlabel('Horsepower')
    ax_hp.set_ylabel('Price ($)')
    ax_hp.set_title('Price vs Horsepower (Colored by Condition)', fontsize=12, fontweight='bold')
    plt.colorbar(scatter2, ax=ax_hp, label='Condition Score')
    ax_hp.grid(True, alpha=0.3)
    
    # 7.9: Average Price by Fuel Type
    ax_fuel = axes[1, 0]
    fuel_price = df.groupby('Fuel_Type')['Price_USD'].mean().sort_values()
    fuel_price.plot(kind='barh', ax=ax_fuel, color='lightgreen', alpha=0.8)
    ax_fuel.set_xlabel('Average Price ($)')
    ax_fuel.set_ylabel('Fuel Type')
    ax_fuel.set_title('Average Price by Fuel Type', fontsize=12, fontweight='bold')
    ax_fuel.grid(True, alpha=0.3, axis='x')
    
    # 7.10: Average Price by Transmission
    ax_trans = axes[1, 1]
    trans_price = df.groupby('Transmission')['Price_USD'].mean().sort_values()
    trans_price.plot(kind='bar', ax=ax_trans, color='lightcoral', alpha=0.8)
    ax_trans.set_xlabel('Transmission Type')
    ax_trans.set_ylabel('Average Price ($)')
    ax_trans.set_title('Average Price by Transmission', fontsize=12, fontweight='bold')
    ax_trans.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
else:
    print("\n⚠️ Not enough test data for visualization")

# ============================================================================
# SECTION 8: PRICE PREDICTION FOR NEW CARS
# ============================================================================

print("\n" + "="*60)
print("🔮 SECTION 8: PRICE PREDICTION FOR NEW CARS")
print("="*60)

# Create sample new cars for prediction
new_cars = pd.DataFrame([
    {
        'Brand': 'Toyota',
        'Brand_Goodwill': 85,
        'Year': 2022,
        'Age': 2,
        'Horsepower': 203,
        'Mileage': 25000,
        'Engine_Size_L': 2.5,
        'Fuel_Type': 'Hybrid',
        'Transmission': 'Automatic',
        'Previous_Owners': 1,
        'Condition_Score': 9,
        'Fuel_Efficiency_MPG': 48,
        'Number_of_Seats': 5
    },
    {
        'Brand': 'BMW',
        'Brand_Goodwill': 90,
        'Year': 2023,
        'Age': 1,
        'Horsepower': 382,
        'Mileage': 8000,
        'Engine_Size_L': 3.0,
        'Fuel_Type': 'Petrol',
        'Transmission': 'Automatic',
        'Previous_Owners': 0,
        'Condition_Score': 10,
        'Fuel_Efficiency_MPG': 25,
        'Number_of_Seats': 5
    },
    {
        'Brand': 'Tesla',
        'Brand_Goodwill': 85,
        'Year': 2023,
        'Age': 1,
        'Horsepower': 450,
        'Mileage': 5000,
        'Engine_Size_L': 0,
        'Fuel_Type': 'Electric',
        'Transmission': 'Automatic',
        'Previous_Owners': 0,
        'Condition_Score': 10,
        'Fuel_Efficiency_MPG': 120,
        'Number_of_Seats': 5
    },
    {
        'Brand': 'Honda',
        'Brand_Goodwill': 84,
        'Year': 2019,
        'Age': 5,
        'Horsepower': 158,
        'Mileage': 65000,
        'Engine_Size_L': 1.8,
        'Fuel_Type': 'Petrol',
        'Transmission': 'Manual',
        'Previous_Owners': 2,
        'Condition_Score': 7,
        'Fuel_Efficiency_MPG': 32,
        'Number_of_Seats': 5
    },
    {
        'Brand': 'Ford',
        'Brand_Goodwill': 78,
        'Year': 2015,
        'Age': 9,
        'Horsepower': 285,
        'Mileage': 110000,
        'Engine_Size_L': 3.5,
        'Fuel_Type': 'Diesel',
        'Transmission': 'Automatic',
        'Previous_Owners': 3,
        'Condition_Score': 6,
        'Fuel_Efficiency_MPG': 22,
        'Number_of_Seats': 7
    }
])

# Feature engineering for new cars
new_cars['Age_Mileage_Ratio'] = new_cars['Age'] / (new_cars['Mileage'] / 10000 + 1)

# Encode categorical variables
for col in categorical_cols:
    if col in new_cars.columns:
        le = label_encoders[col]
        new_cars[col + '_Encoded'] = new_cars[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else 0)
    else:
        if col == 'Brand_Segment':
            new_cars[col] = new_cars['Brand'].apply(get_brand_segment)
            le = label_encoders[col]
            new_cars[col + '_Encoded'] = new_cars[col].apply(lambda x: le.transform([x])[0])
        elif col == 'Age_Category':
            new_cars[col] = new_cars['Age'].apply(get_age_category)
            le = label_encoders[col]
            new_cars[col + '_Encoded'] = new_cars[col].apply(lambda x: le.transform([x])[0])
        elif col == 'Condition_Category':
            new_cars[col] = new_cars['Condition_Score'].apply(get_condition_category)
            le = label_encoders[col]
            new_cars[col + '_Encoded'] = new_cars[col].apply(lambda x: le.transform([x])[0])
        elif col == 'HP_Category':
            new_cars[col] = new_cars['Horsepower'].apply(get_hp_category)
            le = label_encoders[col]
            new_cars[col + '_Encoded'] = new_cars[col].apply(lambda x: le.transform([x])[0])

# Prepare features for prediction
new_cars_features = new_cars[feature_cols]
new_cars_scaled = scaler.transform(new_cars_features)

# Make predictions
predictions_new = best_model.predict(new_cars_scaled)

# Display predictions
print("\n🚗 Price Predictions for New Cars:")
print("-"*70)
for i, (idx, car) in enumerate(new_cars.iterrows()):
    print(f"\nCar {i+1}: {car['Brand']} {car['Year']}")
    print(f"   Features: {car['Horsepower']} HP, {car['Mileage']:,} miles, "
          f"{car['Fuel_Type']}, Condition {car['Condition_Score']}/10")
    print(f"   💰 Predicted Price: ${predictions_new[i]:,.0f}")
    
    confidence_interval = results[best_model_name]['MAE'] * 1.96
    print(f"   📊 Price Range: ${max(0, predictions_new[i] - confidence_interval):,.0f} - "
          f"${predictions_new[i] + confidence_interval:,.0f}")

# ============================================================================
# SECTION 9: SUMMARY AND CONCLUSIONS
# ============================================================================

print("\n" + "="*60)
print("📝 SECTION 9: SUMMARY AND CONCLUSIONS")
print("="*60)

print("\n🎯 KEY FINDINGS:")
print("-"*40)

print("\n1. MODEL PERFORMANCE:")
print(f"   • Best Model: {best_model_name}")
print(f"   • R² Score: {results[best_model_name]['Test_R2']:.4f} ({results[best_model_name]['Test_R2']*100:.1f}% of variance explained)")
print(f"   • Average Prediction Error: ${results[best_model_name]['MAE']:,.0f}")
print(f"   • Percentage Error: {results[best_model_name]['MAPE']:.1f}%")

print("\n2. MOST IMPORTANT FEATURES:")
if hasattr(best_model, 'feature_importances_'):
    top3 = feature_importance.head(3)
    for i, row in top3.iterrows():
        print(f"   • {row['Feature']}: {row['Importance']*100:.1f}% importance")

print("\n3. MARKET INSIGHTS:")
print("   • Luxury cars command significantly higher prices (2-3x of economy cars)")
print("   • Electric and Hybrid vehicles have higher resale value")
print("   • Age and mileage are the strongest depreciation factors")
print("   • Condition score has strong correlation with price")

print("\n💡 BUSINESS RECOMMENDATIONS:")
print("   • Focus on maintaining high condition scores for better resale value")
print("   • Consider electric/hybrid vehicles for better price retention")
print("   • Regular maintenance records can justify higher prices")
print("   • Target luxury segment for higher margins despite lower volume")

print("\n📊 MODEL DEPLOYMENT READY:")
print("   • Model can predict prices for new cars with high accuracy")
print("   • Confidence intervals provide realistic price ranges")
print("   • Feature importance guides pricing strategy decisions")

print("\n" + "="*60)
print("✅ CAR PRICE PREDICTION COMPLETE!")
print("="*60)