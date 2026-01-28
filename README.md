# **Pneumonia Detection using CNN**
*A **Semester 3 MSc. Computer Vision & AI** project at **Berlin School of Business & Innovation (BSBI)**, where I built a **Convolutional Neural Network (CNN)** to classify chest X-ray images for pneumonia detection, achieving **92.31% accuracy**. This project aligns with my professional experience at **Sakon (GSG)**, where I optimized data workflows and reduced errors by **11%**, and my focus on **AI-driven solutions for real-world challenges** like healthcare diagnostics.*

---

## **📌 Project Overview**
This project leverages **deep learning (CNN)** to detect pneumonia from chest X-ray images, addressing a critical healthcare challenge: **early and accurate diagnosis**. Using a dataset of **5,856 X-ray images** (Normal/Pneumonia) from Kaggle, I designed a **4-layer CNN architecture** with data augmentation, class weighting, and learning rate scheduling to achieve **92.31% test accuracy**. The project mirrors my work at **Sakon**, where I automated data processes and improved decision-making through analytics.

### **Key Achievements**
✅ **Data Augmentation**: Applied **shear, zoom, and horizontal flip** to improve model generalization—similar to my **ETL optimization** at Sakon.
✅ **Class Weighting**: Addressed **imbalanced data** (more pneumonia cases) using `compute_class_weight`, ensuring fair model performance.
✅ **Learning Rate Scheduling**: Used `ReduceLROnPlateau` to dynamically adjust learning rates, boosting convergence.
✅ **Model Evaluation**: Achieved **92.31% accuracy** with detailed analysis of precision/recall trade-offs, reflecting my **analytical rigor** at Mercedes-Benz.

---

## **🔧 Technologies & Tools**
   **Tool/Technique**       | **Purpose**                                                                 | **Relevance to My CV**                                  |
 |--------------------------|-----------------------------------------------------------------------------|--------------------------------------------------------|
 | **TensorFlow/Keras**     | Built and trained the CNN model.                                           | Used at **Sakon** for data pipeline automation.       |
 | **Python**               | Data preprocessing, augmentation, and model evaluation (NumPy, Pandas). | Aligns with my **Python-based ETL projects**.         |
 | **ImageDataGenerator**   | Augmented training data (rescaling, shearing, zooming).                     | Mirrors my **data quality improvements** (11% error reduction). |
 | **CNN Architecture**    | 4 convolutional layers + max-pooling for feature extraction.              | Validates my **model design skills** (e.g., time-series at Mercedes-Benz). |
 | **ReduceLROnPlateau**    | Dynamically adjusted learning rates for better convergence.               | Reflects my **process optimization** experience.    |
 | **Matplotlib/Seaborn**    | Visualized model performance (accuracy/loss plots, confusion matrix).      | Matches my **Power BI dashboard** work at Sakon.        |

---

## **📂 Project Structure**

object-detection-yolov11/
│
├── data/
│   ├── train/              # 5,216+ annotated snooker ball images
│   ├── val/                # Validation set
│   └── test/               # Test set for inference
│
├── notebooks/
│   └── YOLOv11_Object_Detection.ipynb  # Colab notebook with full code
│
├── outputs/
│   ├── confusion_matrix.png # Model performance visualization
│   ├── results.png         # Training metrics (mAP, precision/recall)
│   ├── val_predictions/    # Annotated validation images
│   └── test_predictions/   # Annotated test images
│
├── reports/
│   └── Object_Detection_Report.pdf  # Full project documentation
│
└── README.md
Copy

---
## **🚀 Key Insights & CV Parallels**
### **1. YOLOv11 Training = Scalable AI Solutions**
- **Project**: Trained YOLOv11 on snooker balls to simulate **real-world object detection** (e.g., wildlife in satellite imagery).
- **CV Validation**:
  - At **Sakon**, I reduced data errors by **11%** through similar **data cleaning/augmentation** techniques.
  - Demonstrates my ability to **adapt models to new domains** (e.g., environmental monitoring).

### **2. Roboflow Integration = Efficient Data Pipelines**
- **Project**: Used Roboflow for **dataset management and augmentation**, ensuring high-quality training data.
- **CV Validation**:
  - Mirrors my **ETL automation** at Sakon, where I optimized data workflows for **1,200+ customer accounts**.

### **3. Real-Time Inference = Practical AI Applications**
- **Project**: Deployed the model for **real-time snooker ball detection**, with potential extensions to **environmental/entertainment use cases**.
- **CV Validation**:
  - Aligns with my **predictive modeling** work at Mercedes-Benz (e.g., time-series forecasting).

### **4. Model Evaluation = Performance Optimization**
- **Project**: Evaluated using **confusion matrices, mAP scores, and visual annotations**, achieving **99%+ precision for key classes**.
- **CV Validation**:
  - Reflects my **model evaluation** experience (e.g., SVM accuracy analysis in healthcare projects).

---
## **📊 Model Performance**
| **Metric**               | **Score**       | **Interpretation**                                                                 | **Improvement Plan**                          |
|--------------------------|-----------------|------------------------------------------------------------------------------------|-----------------------------------------------|
| **Precision (Ball Classes)** | 99%+        | Model excels at localizing snooker balls with minimal false positives.          | Test on **diverse datasets** (e.g., wildlife). |
| **Recall (All Classes)**  | 90%+            | High detection rate across classes.                                              | Add **more augmented data** for edge cases.    |
| **mAP@0.5**              | 0.75            | Strong average precision across IoU threshold.                                   | Experiment with **YOLOv11-large** for higher mAP. |
| **Inference Speed**      | ~60ms/image     | Real-time capable on GPU.                                                          | Optimize for **edge devices** (e.g., Jetson Nano). |

**Key Takeaway**:
The model demonstrates **high accuracy in controlled settings** (snooker balls) and can be extended to **real-world applications** (e.g., environmental monitoring) with additional training data.

---
## **🛠 How to Run This Project**
1. **Prerequisites**:
   - Python 3.8+
   - Libraries: `ultralytics`, `roboflow`, `supervision`, `torch`
     ```bash
     pip install ultralytics roboflow supervision torch
     ```

2. **Steps**:
   ```bash
   git clone https://github.com/yourusername/object-detection-yolov11.git
   cd object-detection-yolov11/notebooks
   jupyter notebook YOLOv11_Object_Detection.ipynb


Follow the notebook to:

Load the snooker ball dataset via Roboflow.
Train YOLOv11 with ReduceLROnPlateau callback.
Evaluate using confusion matrices and test predictions.


Expected Output:

Annotated Images: Bounding boxes on snooker balls (validation/test sets).
Performance Plots: mAP, precision/recall curves.
Confusion Matrix: Per-class detection accuracy.


🌟 Future Enhancements

Environmental Monitoring: Adapt the model to detect wildlife/vehicles in satellite imagery.
Entertainment AI: Extend to gesture/expression recognition for social media apps.
Edge Deployment: Optimize for real-time inference on IoT devices (e.g., drones).
Multi-Object Tracking: Integrate SORT/DeepSORT for dynamic scenes.

📄 License
This project is open-source under the MIT License.

---

For a visual walkthrough of the project outputs, check out the OutputSnaps folder—I’ve compiled chronologically arranged snapshots of all key visualizations and results for easy reference! 📸📂
---
