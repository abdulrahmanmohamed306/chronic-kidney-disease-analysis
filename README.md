# Chronic Kidney Disease Prediction & Analysis Project

مشروع تحليلي متكامل لتشخيص والتنبؤ بمرض الفشل الكلوي المزمن (Chronic Kidney Disease) باستخدام تقنيات علم البيانات وتعلم الآلة (Machine Learning).

## 📊 نظرة عامة على المشروع
يهدف هذا المشروع إلى تنظيف وتحليل بيانات المرضى الطبية، استكشاف الأنماط المؤثرة، وبناء نموذج ذكاء اصطناعي قادر على التنبؤ بالإصابة بدقة عالية بناءً على المؤشرات الحيوية والمخبرية.

## 🛠️ مراحل العمل الرئيسية
1. **تنظيف البيانات (Data Cleaning & Imputation):**
   - التعامل مع القيم المفقودة بكفاءة عالية (صفر مفقودات نهائياً).
   - استخدام `KNNImputer` للأعمدة الرقمية و `Mode` للأعمدة النصية والفئوية.
2. **التحليل الاستكشافي للبيانات (EDA & Visualization):**
   - رسم وتوزيع الحالات (مصاب `ckd` مقابل سليم `notckd`).
   - دراسة الارتباطات بين المؤشرات الطبية باستخدام `Correlation Heatmap`.
   - رسم مخططات التوزيع والعلاقات (`Scatter Plots`, `Box Plots`, `Pair Plots`).
3. **التجهيز والنمذجة (Preprocessing & Machine Learning):**
   - ترميز المتغيرات النصية باستخدام `LabelEncoder`.
   - توحيد المقاييس الرقمية باستخدام `StandardScaler`.
   - تطبيق نموذج **Random Forest Classifier**.
4. **التقييم والتحقق (Evaluation & Validation):**
   - تحقيق دقة (`Accuracy`) بنسبة **100.00%** على بيانات الاختبار.
   - إجراء **Cross-Validation (5-Folds)** لتأكيد ثبات وكفاءة النموذج بنسبة استقرار 100%.

## 🔍 أهم المؤشرات الطبية المؤثرة (Top Feature Importances)
أظهر نموذج الـ Random Forest أن أهم المتغيرات الطبية التي تم الاعتماد عليها في التشخيص هي:
- **`sg`** (الوزن النوعي للبول - Specific Gravity)
- **`hemo`** (الهيموجلوبين - Hemoglobin)
- **`pcv`** (حجم الخلايا المتراصة - Packed Cell Volume)
- **`sc`** (كرياتينين السيروم - Serum Creatinine)
- **`rc`** (كريات الدم الحمراء - Red Blood Cell Count)

## ⚙️ كيفية تشغيل المشروع
قم بتثبيت المكتبات المطلوبة أولاً:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn