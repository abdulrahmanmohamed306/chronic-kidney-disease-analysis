# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 13:46:46 2026

@author: sales
"""

import pandas as pd
import numpy as np

# قراءة الملف من المسار
df = pd.read_csv(r'C:\Users\sales\Desktop\kidney_disease\kidney_disease.csv')

# عرض أول 5 صفوف للتأكد أن الداتا قرأت تمام
print(df.head())

# عرض معلومات سريعة عن شكل البيانات وأبعادها
print("Shape of data:", df.shape)
# حساب عدد النسبة المئوية للقيم المفقودة لكل عمود
missing_data = pd.DataFrame({
    'Missing Count': df.isnull().sum(),
    'Missing Percentage (%)': (df.isnull().sum() / len(df)) * 100
})

# طباعة الأعمدة التي تحتوي على قيم مفقودة فقط وترتيبها تنازلياً
print(missing_data[missing_data['Missing Count'] > 0].sort_values(by='Missing Count', ascending=False))
from sklearn.impute import KNNImputer

# اختيار الأعمدة الرقمية فقط للتعويض المتقدم
num_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

# تطبيق KNN Imputer (مثلاً بالاعتماد على أقرب 5 حالات مشابهة)
imputer = KNNImputer(n_neighbors=5)
df[num_cols] = imputer.fit_transform(df[num_cols])

print("تم تعويض الفراغات الرقمية بنجاح، إجمالي المفقودات الآن:", df.isnull().sum().sum())
from sklearn.impute import SimpleImputer
import numpy as np

# تنظيف المسافات والـ nan في الأعمدة النصية أولاً
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].str.strip()
    df[col] = df[col].replace({'nan': np.nan, 'NaN': np.nan})

# تطبيق تعويض المنوال (الأكثر تكراراً) للأعمدة النصية
cat_cols = df.select_dtypes(include=['object']).columns.tolist()
cat_imputer = SimpleImputer(strategy='most_frequent')
df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

# التأكد من خلو الجدول تماماً من أي قيم مفقودة
print("Total missing values in entire dataset now:", df.isnull().sum().sum())
import matplotlib.pyplot as plt
import seaborn as sns

# ضبط شكل الرسم البياني
plt.figure(figsize=(6, 4))
sns.countplot(x='classification', data=df, palette='Set2')

# إضافة العناوين والتفاصيل
plt.title('Distribution of Kidney Disease Classification', fontsize=12)
plt.xlabel('Classification', fontsize=10)
plt.ylabel('Count', fontsize=10)

# عرض الرسم
plt.show()
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. الرسمة الأولى: Scatter Plot (العمر مقابل ضغط الدم) ---
plt.figure(figsize=(8, 5))
sns.scatterplot(x='age', y='bp', hue='classification', data=df, palette='Set1', alpha=0.8)
plt.title('Relationship between Age, Blood Pressure and Classification', fontsize=12)
plt.xlabel('Age', fontsize=10)
plt.ylabel('Blood Pressure (bp)', fontsize=10)
plt.legend(title='Classification')
plt.show()

# --- 2. الرسمة الثانية: Correlation Heatmap (خريطة الارتباط) ---
plt.figure(figsize=(12, 10))
# استبعاد عمود الـ id لو موجود عشان ما يؤثرش على الحسابات الرقمية
numeric_df = df.drop(columns=['id']) if 'id' in df.columns else df
corr_matrix = numeric_df.select_dtypes(include=['float64', 'int64']).corr()

sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5, cbar=True)
plt.title('Correlation Heatmap of Numerical Features', fontsize=14)
plt.show()
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Box Plots (مخطط الصندوق: كرياتينين الدم مقابل الإصابة) ---
plt.figure(figsize=(7, 5))
sns.boxplot(x='classification', y='sc', data=df, palette='Set2')
plt.title('Serum Creatinine (sc) Distribution by Classification', fontsize=12)
plt.xlabel('Classification', fontsize=10)
plt.ylabel('Serum Creatinine (sc)', fontsize=10)
plt.show()

# --- 2. Categorical Bar Plots (مخططات الأعمدة للأمراض المزمنة: ضغط الدم والسكر) ---
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.countplot(x='htn', hue='classification', data=df, palette='Set2')
plt.title('Hypertension (htn) vs Classification', fontsize=11)
plt.xlabel('Hypertension (yes/no)')
plt.ylabel('Count')

plt.subplot(1, 2, 2)
sns.countplot(x='dm', hue='classification', data=df, palette='Set2')
plt.title('Diabetes Mellitus (dm) vs Classification', fontsize=11)
plt.xlabel('Diabetes (yes/no)')
plt.ylabel('Count')

plt.tight_layout()
plt.show()

# --- 3. Histograms & KDE (مخططات التوزيع التكراري: العمر وسكر الدم العشوائي) ---
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.histplot(df['age'], kde=True, color='teal', bins=20)
plt.title('Age Distribution (KDE)', fontsize=11)
plt.xlabel('Age')

plt.subplot(1, 2, 2)
sns.histplot(df['bgr'], kde=True, color='orange', bins=20)
plt.title('Blood Glucose Random (bgr) Distribution', fontsize=11)
plt.xlabel('Blood Glucose Random')

plt.tight_layout()
plt.show()

# --- 4. Pair Plot (مخطط الأزواج لأهم المؤشرات الحيوية) ---
# اختيار أهم الأعمدة الرقمية مع التصنيف لتجنب ازدحام الرسمة
selected_cols = ['age', 'bp', 'sc', 'hemo', 'classification']
sns.pairplot(df[selected_cols], hue='classification', palette='Set1', diag_kind='kde', height=2.2)
plt.show()
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# نسخة احتياطية للعمل عليها
df_model = df.copy()

# 1. ترميز المتغيرات النصية إلى أرقام
le = LabelEncoder()
for col in df_model.select_dtypes(include=['object']).columns:
    df_model[col] = le.fit_transform(df_model[col])

# 2. فصل المدخلات (Features - X) عن الهدف (Target - y)
# عمود 'classification' هو الهدف (0 أو 1)
X = df_model.drop(columns=['classification', 'id'] if 'id' in df_model.columns else ['classification'])
y = df_model['classification']

# 3. تقسيم الداتا إلى تدريب (Train) واختبار (Test) بنسبة 80% لـ 20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. توحيد المقاييس (Feature Scaling) لضمان دقة الخوارزميات
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Shape of X_train:", X_train_scaled.shape)
print("Shape of X_test:", X_test_scaled.shape)
print("تم تجهيز وتقسيم البيانات بنجاح وجاهزون للتدريب!")
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 1. بناء وتدريب النموذج
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_scaled, y_train)

# 2. التنبؤ على بيانات الاختبار
y_pred = rf_model.predict(X_test_scaled)

# 3. حساب دقة النموذج (Accuracy)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# 4. طباعة تقرير الأداء التفصيلي (Classification Report)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 5. رسم مصفوفة الارتباك (Confusion Matrix)
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix', fontsize=12)
plt.xlabel('Predicted Label', fontsize=10)
plt.ylabel('True Label', fontsize=10)
plt.show()
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# استخراج أهمية الميزات من نموذج Random Forest
feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

# طباعة أعلى 10 أعمدة مؤثرة
print("Top 10 Most Important Features:")
print(feature_importances.head(10))

# رسم النتائج بياناً
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importances.head(10), palette='viridis')
plt.title('Top 10 Feature Importances in Kidney Disease Prediction', fontsize=12)
plt.xlabel('Importance Score', fontsize=10)
plt.ylabel('Features', fontsize=10)
plt.show()
from sklearn.model_selection import cross_val_score

# تطبيق Cross-Validation على الداتا الموحدة (5 أجزاء)
cv_scores = cross_val_score(rf_model, scaler.fit_transform(X), y, cv=5)

print("Cross-Validation Scores for all folds:", cv_scores)
print(f"Mean Cross-Validation Accuracy: {cv_scores.mean() * 100:.2f}%")