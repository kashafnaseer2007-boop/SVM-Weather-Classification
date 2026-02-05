import joblib
from datetime import datetime

sample_weather = {
    'Temperature': 18,
    'Humidity': 65,
    'Wind Speed': 12,
    'Precipitation (%)': 85,
    'Cloud Cover': 'overcast',
    'Atmospheric Pressure': 1005,
    'UV Index': 2,
    'Season': 'Autumn',
    'Visibility (km)': 4,
    'Location': 'mountain'
}

# Convert to dataframe
sample_df = pd.DataFrame([sample_weather])

preprocessor = joblib.load('/content/weather_preprocessor_20260204_1123.pkl')
svm_model = joblib.load('/content/weather_svm_model_20260204_1123.pkl')

# Use your saved model (if you ran the save cell)
try:
    processed_sample = preprocessor.transform(sample_df)
    prediction = svm_model.predict(processed_sample)
    print(f"🌤️  Prediction for sample weather: {prediction}")
    print("(Should probably be 'Rainy' with those numbers!)")
except:
    print("⚠️ Run the save cell first to use this!")

# One last cool visualization - prediction confidence
decision_scores = svm_model.decision_function(processed_sample)
weather_classes = svm_model.classes_

print("🏆 Model Confidence Scores:")
for weather, score in zip(weather_classes, decision_scores[0]):
    print(f"  {weather:10} → {score:+.3f}")

# Highest score wins!
winner = weather_classes[np.argmax(decision_scores)]
print(f"\n🎯 Highest confidence: {winner} (score: {np.max(decision_scores):.3f})")
