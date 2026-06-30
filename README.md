
# 📊 تحلیل احساسات نظرات یوتیوب

یک وب‌اپلیکیشن کاملاً تعاملی ساخته‌شده با **Streamlit** برای دانلود نظرات یوتیوب و تحلیل احساسات آن‌ها با استفاده از چندین مدل NLP شامل:

* **Twitter RoBERTa (HuggingFace)**
* **مدل هیبریدی (LR-VADER → در صورت عدم اطمینان RoBERTa)**
* **مدل Logistic Regression + ویژگی‌های VADER**
* **مدل Linear SVM + ویژگی‌های VADER**
* **مدل‌های سفارشی ذخیره‌شده با joblib**

---

## قابلیت‌های اپلیکیشن

✓ استخراج خودکار نظرات یوتیوب
✓ امکان آپلود فایل CSV
✓ مقایسه چند مدل مختلف NLP
✓ تشخیص زبان و فیلتر کردن آن
✓ مدل هیبریدی مبتنی بر confidence
✓ نمایش ماتریس درهم‌ریختگی (Confusion Matrix)
✓ دانلود خروجی نتایج

---

## 🚀 ویژگی‌ها

### 🔹 1. دریافت نظرات یوتیوب

با وارد کردن لینک ویدیو + API Key می‌توانید تا **۵۰۰۰ کامنت** را از طریق YouTube Data API دریافت کنید.

---

### 🔹 2. آپلود CSV

می‌توانید یک فایل `.csv` که شامل ستون `Comment` است آپلود کنید تا تحلیل آفلاین انجام شود.

---

### 🔹 3. مدل‌های مختلف NLP

می‌توانید تحلیل احساسات را با مدل‌های زیر انجام دهید:

| مدل                   | توضیح                                                     |
| --------------------- | --------------------------------------------------------- |
| **Twitter-RoBERTa**   | مدل ترنسفورمر از HuggingFace (بالاترین دقت)               |
| **Hybrid Model**      | ترکیب LR + VADER و در صورت عدم اطمینان استفاده از RoBERTa |
| **LR-VADER Models**   | رگرسیون لجستیک با ویژگی‌های VADER                         |
| **Linear SVM Models** | مدل SVM با ویژگی‌های VADER                                |
| **All Models**        | اجرای همه مدل‌ها و مقایسه آن‌ها                           |

---

### 🔹 4. تشخیص زبان

با استفاده از کتابخانه `langdetect`، کامنت‌های غیرانگلیسی شناسایی و برچسب‌گذاری می‌شوند.

---

### 🔹 5. مراحل پیش‌پردازش (Preprocessing Pipeline)

شامل موارد زیر است:

* حذف و نرمال‌سازی URL و mentionها
* تبدیل ایموجی‌ها (emoji demojization)
* Lemmatization
* استخراج ویژگی‌های VADER
* پاک‌سازی با Regex
* استفاده از tokenizer مربوط به HuggingFace

---

### 🔹 6. خروجی قابل دانلود

هر تب مربوط به مدل شامل موارد زیر است:

* نمودار دایره‌ای احساسات
* نمایش تصمیم مدل در حالت هیبریدی
* ماتریس درهم‌ریختگی (Confusion Matrix)
* دکمه دانلود فایل `.csv`

---

## نحوه عملکرد سامانه

![نحوه عملکرد سامانه](Diagram.png)
---
## 📂 ساختار پروژه

```text
project/
│
├── app.py                  # اپلیکیشن اصلی Streamlit
├── final_lr_sentiment_model.joblib
├── more_data_lr_model.joblib
├── final_linear_svm_model.joblib
├── linear_svm_model.joblib
└── roberta-local/          # (اختیاری) مدل RoBERTa دانلودشده به صورت محلی
```

---

## 🔧 راه‌اندازی پروژه

### 1. نصب وابستگی‌ها

```bash
pip install -r requirements.txt
```

---

### 2. اجرای اپلیکیشن

```bash
streamlit run app.py
```

---

## 🔑 تنظیم API یوتیوب

برای استفاده از حالت API:

1. ساخت API Key در Google Cloud Console
2. فعال‌سازی **YouTube Data API v3**
3. وارد کردن API Key در رابط کاربری

---

## 🧠 نحوه عملکرد مدل هیبریدی

1. اجرای Logistic Regression + VADER
2. اگر میزان اطمینان (confidence) کم باشد، ارسال متن به RoBERTa
3. ترکیب نتایج
4. خروجی شامل:

   * احساس (Sentiment)
   * میزان اطمینان
   * اینکه کدام مدل تصمیم گرفته است

این روش باعث افزایش دقت و کاهش هزینه استفاده از RoBERTa می‌شود.

---

## 📈 مقایسه مدل‌ها

زمانی که چند مدل اجرا می‌شوند، سیستم به صورت خودکار:

* هر مدل را با RoBERTa مقایسه می‌کند
* میزان توافق (agreement score) را محاسبه می‌کند
* ماتریس درهم‌ریختگی را برای کامنت‌های انگلیسی تولید می‌کند

---

## 🛡 مدیریت خطا و تجربه کاربری

* fallback در صورت نبود HuggingFace یا GPU یا langdetect
* استفاده از spinner و progress bar در Streamlit
* نمایش هشدارهای واضح برای کاربر
* لاگ‌گیری از خطاها به صورت robust

---
 
# 📊 YouTube Sentiment Analyzer

A fully interactive **Streamlit web app** for downloading YouTube comments and analyzing their sentiment using multiple NLP models including:

* **Twitter RoBERTa (HuggingFace)**
* **Hybrid Model (LR-VADER → RoBERTa fallback)**
* **Logistic Regression + VADER features**
* **Linear SVM + VADER features**
* **Custom models saved via joblib**

The app supports:
✓ Automatic **YouTube comment extraction**
✓ **CSV upload** mode
✓ **Multimodel comparison**
✓ **Language detection & filtering**
✓ Confidence-based hybrid modeling
✓ Confusion matrix visualization
✓ Downloadable results

---

## 🚀 Features

### 🔹 1. Fetch YouTube Comments

Input a video URL + API key → fetch up to **5000 comments** via the YouTube Data API.

### 🔹 2. Upload CSV

Upload a `.csv` containing a `Comment` column to analyze offline data.

### 🔹 3. Multiple NLP Models

You can run sentiment analysis using:

| Model                 | Description                                 |
| --------------------- | ------------------------------------------- |
| **Twitter-RoBERTa**   | HuggingFace transformer (best accuracy)     |
| **Hybrid Model**      | LR-VADER → RoBERTa for low-confidence cases |
| **LR-VADER Models**   | Logistic Regression with VADER features     |
| **Linear SVM Models** | SVM with VADER features                     |
| **All Models**        | Run and compare everything                  |

### 🔹 4. Language Detection

Uses `langdetect` to label non-English comments automatically.

### 🔹 5. Preprocessing Pipeline

Includes:

* URL / mention normalization
* Emoji demojization
* Lemmatization
* VADER feature extraction
* Regex cleanup
* HuggingFace tokenizer support

### 🔹 6. Downloadable Results

Each model tab includes:

* Sentiment pie chart
* Responsible model breakdown (for hybrid mode)
* Confusion matrix comparison
* Download `.csv` button

---

## Diagram For How the System works

![Diagram](Diagram.png)

---

## 📂 Folder Structure

```
project/
│
├── app.py                  # The main Streamlit app
├── final_lr_sentiment_model.joblib
├── more_data_lr_model.joblib
├── final_linear_svm_model.joblib
├── linear_svm_model.joblib
└── roberta-local/          # (optional) auto-downloaded RoBERTa model
```

---

## 🔧 Setup Instructions

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the app

```bash
streamlit run app.py
```

---

## 🔑 YouTube API Setup

To use API mode:

1. Create an API key from Google Cloud Console
2. Enable **YouTube Data API v3**
3. Paste the API key into the UI

---

## 🧠 How the Hybrid Model Works

1. Run Logistic Regression + VADER
2. If **confidence < threshold**, send the text to RoBERTa
3. Merge results
4. Provide:

   * Sentiment
   * Confidence
   * Which model made the decision

This gives high accuracy while keeping RoBERTa costs low.

---

## 📈 Model Comparison

When multiple models run, the app automatically:

* Compares each model to **RoBERTa**
* Calculates agreement score
* Builds a confusion matrix (English-only comments)

---

## 🛡 Error Handling & UX Features

* Graceful fallback if HuggingFace, GPU, or langdetect is missing
* Streamlit spinners and progress bars
* Clear user warnings
* Robust exception logging
